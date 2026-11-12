"""Resumable per-shard SQLite index for the full 20-second WOMD source."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from src.scenarios.womd20 import parse_scenario, raw_records, validate_scenario


INDEX_VERSION = "1"


def scan_index(root, output, limit_shards=None):
    root, output = Path(root).resolve(), Path(output)
    expected = [root / f"training_20s.tfrecord-{i:05d}-of-01000" for i in range(1000)]
    selected = expected if limit_shards is None else expected[:limit_shards]
    if not root.is_dir():
        raise FileNotFoundError(root)
    output.mkdir(parents=True, exist_ok=True)
    db_path = output / "index.sqlite"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS shards (
                source_shard TEXT PRIMARY KEY, size_bytes INTEGER, mtime_ns INTEGER,
                status TEXT, records INTEGER, errors INTEGER, error TEXT);
            CREATE TABLE IF NOT EXISTS scenarios (
                source_shard TEXT, record_index INTEGER, scenario_id TEXT,
                frames INTEGER, duration_s REAL, tracks INTEGER, map_features INTEGER,
                signal_frames INTEGER, current_time_index INTEGER,
                sdc_valid_frames INTEGER, error TEXT,
                PRIMARY KEY (source_shard, record_index));
            CREATE INDEX IF NOT EXISTS scenario_id_lookup ON scenarios(scenario_id);
        """)
        for key, value in (("root", str(root)), ("index_version", INDEX_VERSION)):
            old = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
            if old is not None and old[0] != value:
                raise ValueError(f"Index {key} mismatch; use a new output directory")
            connection.execute("INSERT OR IGNORE INTO metadata VALUES (?, ?)", (key, value))
        connection.commit()
        for number, path in enumerate(selected, 1):
            stat = path.stat() if path.is_file() else None
            previous = connection.execute("SELECT * FROM shards WHERE source_shard=?", (path.name,)).fetchone()
            if (previous and stat and previous["status"] == "ok"
                    and previous["size_bytes"] == stat.st_size and previous["mtime_ns"] == stat.st_mtime_ns):
                print(f"[{number}/{len(selected)}] resume: {path.name}", flush=True)
                continue
            # Commit a whole shard. Interruption rolls back its partial rows.
            with connection:
                connection.execute("DELETE FROM scenarios WHERE source_shard=?", (path.name,))
                count, errors, file_error = 0, 0, ""
                try:
                    if stat is None:
                        raise FileNotFoundError(path)
                    for index, raw in enumerate(raw_records(path)):
                        count += 1
                        row = dict(source_shard=path.name, record_index=index, scenario_id=None,
                                   frames=None, duration_s=None, tracks=None, map_features=None,
                                   signal_frames=None, current_time_index=None, sdc_valid_frames=None, error="")
                        try:
                            scenario = parse_scenario(raw)
                            row["scenario_id"] = scenario.scenario_id or None
                            times = validate_scenario(scenario)
                            row.update(frames=len(times), duration_s=float(times[-1] - times[0]),
                                       tracks=len(scenario.tracks), map_features=len(scenario.map_features),
                                       signal_frames=sum(bool(s.lane_states) for s in scenario.dynamic_map_states),
                                       current_time_index=scenario.current_time_index,
                                       sdc_valid_frames=sum(s.valid for s in scenario.tracks[scenario.sdc_track_index].states))
                        except Exception as exc:
                            errors += 1
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        connection.execute("""INSERT INTO scenarios VALUES
                            (:source_shard, :record_index, :scenario_id, :frames, :duration_s,
                             :tracks, :map_features, :signal_frames, :current_time_index,
                             :sdc_valid_frames, :error)""", row)
                    if count == 0:
                        raise ValueError("Shard has no TFRecord records")
                except Exception as exc:
                    file_error = f"{type(exc).__name__}: {exc}"
                status = "ok" if not file_error and not errors else "error"
                connection.execute("INSERT OR REPLACE INTO shards VALUES (?, ?, ?, ?, ?, ?, ?)",
                                   (path.name, stat.st_size if stat else 0, stat.st_mtime_ns if stat else 0,
                                    status, count, errors, file_error))
            print(f"[{number}/{len(selected)}] {status}: {path.name} records={count} errors={errors} {file_error}", flush=True)

        # Exports are replaced atomically; SQLite remains the resume source.
        for table, filename in (("scenarios", "scenarios.csv"), ("shards", "shards.csv")):
            cursor = connection.execute(f"SELECT * FROM {table} ORDER BY source_shard" + (", record_index" if table == "scenarios" else ""))
            temporary = output / (filename + ".tmp")
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow([col[0] for col in cursor.description])
                writer.writerows(cursor)
            temporary.replace(output / filename)
        summary = dict(connection.execute("""SELECT COUNT(*) AS scenarios,
            SUM(CASE WHEN error != '' THEN 1 ELSE 0 END) AS invalid_records,
            MIN(duration_s) AS min_duration_s, MAX(duration_s) AS max_duration_s,
            MIN(frames) AS min_frames, MAX(frames) AS max_frames FROM scenarios""").fetchone())
        shard_rows = connection.execute("SELECT * FROM shards").fetchall()
        duplicate_ids = connection.execute("""SELECT scenario_id, COUNT(*) AS occurrences FROM scenarios
            WHERE scenario_id IS NOT NULL GROUP BY scenario_id HAVING COUNT(*) > 1""").fetchall()
        summary.update(source_root=str(root), index_version=INDEX_VERSION,
                       indexed_shards=len(shard_rows),
                       failed_shards=sum(s["status"] != "ok" for s in shard_rows),
                       missing_source_shards=[p.name for p in expected if not p.is_file()],
                       duplicate_scenario_ids=[dict(row) for row in duplicate_ids],
                       validation_scope="TFRecord CRC, protobuf parse, timestamp/track/signal structure; not full semantic validation")
        summary["full_scan_ok"] = (len(shard_rows) == 1000 and not summary["failed_shards"]
                                   and not summary["missing_source_shards"] and not duplicate_ids)
        temporary = output / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "summary.json")
        print(f"Index: {db_path}; scenarios={summary['scenarios']}; full_scan_ok={summary['full_scan_ok']}", flush=True)
        return summary
    finally:
        connection.close()
