"""Research-grade, interaction-aware MERGE classification (schema v2).

Unlike :mod:`src.scenarios.merge_detector`, this module never infers map
connectivity from truncated roadgraph polylines.  Its topology input must come
from a WOMD ``Scenario`` protobuf (``LaneCenter.entry_lanes/exit_lanes``).
Future trajectory is allowed only here, in the offline labelling pipeline; the
online policy observation must not import or call this module.
"""

import dataclasses
import enum
from typing import Optional, Tuple


# Historical dataset-construction schema tags. Only ManeuverSpec.
# from_csv_row (legacy-format CSV loader, still used to reconstruct
# the intermediate candidate pool feeding the final dataset's own
# construction chain -- see scripts/build_merge_manifest.py/
# extract_merge_scenes.py) and ManeuverSpec.require_schema's matching
# branches ever assign/check these; the final PPO/visualization
# runtime path uses MERGE_DATASET_SCHEMA exclusively.
DATASET_SCHEMA_V2 = "merge_interaction_v2"
LEGACY_DATASET_SCHEMA = "merge_geometry_v1_legacy"

MERGE_DATASET_SCHEMA = "merge_decision"
"""The final MERGE Dataset's PPO decision-context contract
(data/manifests/merge_*.csv, data/manifests/freeze.json). Selects
episodes by Merge Context Validity + Decision Relevance (a maneuver is
a valid PPO decision context -- e.g. a KEEP-only WEAK_INTERACTION case
with no interaction vehicle at all -- without ever having been
manually reviewed); deliberately does NOT require manual_validation ==
CONFIRMED_MERGE or a relevant interaction vehicle, unlike the
historical DATASET_SCHEMA_V2 contract above."""


class ManeuverType(enum.Enum):
    TOPOLOGICAL_MERGE = "topological_merge"
    INTERACTIVE_CUT_IN = "interactive_cut_in"
    ROUNDABOUT_ENTRY = "roundabout_entry"
    NON_INTERACTIVE_MERGE = "non_interactive_merge"


class V2Decision(enum.Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REVIEW = "review"


REJECT_SERIAL_CONTINUATION = "serial_continuation"
REJECT_LANE_CHANGE = "lane_change"
REJECT_CUT_IN_NOT_TOPOLOGY_MERGE = "cut_in_not_topology_merge"
REJECT_INTERSECTION = "intersection_transition"
REJECT_DIVERGE = "diverge_split"
REJECT_MAP_ARTIFACT = "map_artifact"
REJECT_NO_AUTHORITATIVE_TYPE = "no_authoritative_merge_type"
REJECT_INSUFFICIENT_SOURCE_HISTORY = "insufficient_source_history"
REJECT_NO_PHYSICAL_TRANSITION = "no_physical_transition"
REJECT_TARGET_NOT_STABLE = "target_not_stable"
REJECT_NO_INTERACTION = "no_relevant_vehicle_interaction"
REJECT_GAP_ORDER_NOT_PERSISTENT = "gap_order_not_persistent"
REVIEW_AMBIGUOUS_TOPOLOGY = "ambiguous_topology_requires_manual_review"


@dataclasses.dataclass(frozen=True)
class TopologyEvidence:
    """Authoritative topology plus geometry needed to classify one event."""

    source_lane_id: int
    target_lane_id: int
    source_exit_lane_ids: Tuple[int, ...]
    target_entry_lane_ids: Tuple[int, ...]
    source_exits_to_target: bool
    target_is_roundabout: bool = False
    lanes_overlap_longitudinally: bool = False
    terminal_heading_difference_deg: float = 180.0
    terminal_collinear_offset_m: float = float("inf")
    endpoint_gap_m: float = float("inf")
    # Additional signals needed to tell topological merge apart from
    # lane-change / intersection / diverge / map-artifact false positives.
    # Optional/None means "not computed by this caller yet" -- absence
    # must never be silently treated as a negative signal (see
    # ``classify_v2_merge``'s REVIEW fallback).
    is_lane_change_target: Optional[bool] = None
    is_intersection_transition: Optional[bool] = None
    source_exit_lane_count: Optional[int] = None
    target_polyline_point_count: Optional[int] = None

    @property
    def target_has_multiple_entries(self) -> bool:
        return len(set(self.target_entry_lane_ids)) >= 2

    @property
    def is_single_path_continuation(self) -> bool:
        """True for a sampled-map segment boundary, even with a 1--5 m gap."""

        return (
            self.source_exits_to_target
            and set(self.target_entry_lane_ids) == {self.source_lane_id}
            and not self.lanes_overlap_longitudinally
            and self.terminal_heading_difference_deg <= 5.0
            and self.terminal_collinear_offset_m <= 1.0
            and self.endpoint_gap_m <= 5.0
        )

    @property
    def is_diverge(self) -> bool:
        """Source lane splits into multiple exits and ego took a path
        that is not the target of this candidate -- a fork, not a merge.
        Requires at least 2 distinct exits so a merge target that also
        happens to be the source's sole exit is never misclassified."""

        return (
            self.source_exit_lane_count is not None
            and self.source_exit_lane_count >= 2
            and not self.target_has_multiple_entries
            and not self.lanes_overlap_longitudinally
        )

    @property
    def is_map_artifact(self) -> bool:
        """A target lane with a degenerate polyline (0-1 points) cannot
        be a real driveable path; treat convergence onto it as a map
        sampling artifact rather than a real merge."""

        return (
            self.target_polyline_point_count is not None
            and self.target_polyline_point_count <= 1
        )


@dataclasses.dataclass(frozen=True)
class InteractionEvidence:
    """Offline temporal evidence. IDs/gaps are computed over the decision window."""

    decision_start_frame: int
    commit_frame: int
    completion_frame: int
    source_occupancy_frames: int
    target_stable_frames: int
    longitudinal_progress_m: float
    lateral_displacement_m: float
    front_vehicle_id: Optional[int] = None
    rear_vehicle_id: Optional[int] = None
    conflict_vehicle_ids: Tuple[int, ...] = ()
    gap_order_valid_at_entry: bool = False
    gap_order_persistence_frames: int = 0
    required_gap_order_persistence_frames: int = 5

    @property
    def relevant_vehicle_ids(self) -> Tuple[int, ...]:
        ids = list(self.conflict_vehicle_ids)
        if self.front_vehicle_id is not None:
            ids.append(self.front_vehicle_id)
        if self.rear_vehicle_id is not None:
            ids.append(self.rear_vehicle_id)
        return tuple(dict.fromkeys(ids))


@dataclasses.dataclass(frozen=True)
class V2MergeDiagnostic:
    decision: V2Decision
    reason: Optional[str]
    maneuver_type: Optional[ManeuverType]
    topology: TopologyEvidence
    interaction: InteractionEvidence
    interactive: bool
    canonical: bool


def classify_v2_merge(
    topology: TopologyEvidence,
    interaction: InteractionEvidence,
    *,
    min_source_occupancy_frames: int = 5,
    min_target_stable_frames: int = 5,
    min_longitudinal_progress_m: float = 0.1,
    min_cut_in_lateral_displacement_m: float = 1.0,
) -> V2MergeDiagnostic:
    """Apply the frozen historical v2 hard gates in fixed order.

    ``ACCEPT`` here means accepted under the v2 construction contract, not a
    physically confirmed on-ramp merge. New audits and success artifacts must
    additionally use :mod:`src.scenarios.physical_merge_validation`.
    """

    def result(decision, reason, maneuver_type=None, interactive=False):
        return V2MergeDiagnostic(
            decision=decision,
            reason=reason,
            maneuver_type=maneuver_type,
            topology=topology,
            interaction=interaction,
            interactive=interactive,
            canonical=(decision == V2Decision.ACCEPT and interactive),
        )

    if topology.is_single_path_continuation:
        return result(V2Decision.REJECT, REJECT_SERIAL_CONTINUATION)

    if topology.is_map_artifact:
        return result(V2Decision.REJECT, REJECT_MAP_ARTIFACT)

    if topology.is_diverge:
        return result(V2Decision.REJECT, REJECT_DIVERGE)

    if topology.is_lane_change_target:
        return result(V2Decision.REJECT, REJECT_LANE_CHANGE)

    if topology.is_intersection_transition:
        return result(V2Decision.REJECT, REJECT_INTERSECTION)

    # A pure lateral-overlap "cut-in" with no map-topology merge (no
    # entry_lanes/exit_lanes convergence, not a roundabout) is a lane
    # change or interactive cut-in that never actually merges by the
    # map -- excluded from the MERGE definition even when interactive.
    if (
        not topology.source_exits_to_target
        and topology.lanes_overlap_longitudinally
        and interaction.lateral_displacement_m >= min_cut_in_lateral_displacement_m
    ):
        return result(V2Decision.REJECT, REJECT_CUT_IN_NOT_TOPOLOGY_MERGE)

    if topology.target_is_roundabout and topology.source_exits_to_target:
        maneuver_type = ManeuverType.ROUNDABOUT_ENTRY
    elif topology.source_exits_to_target and topology.target_has_multiple_entries:
        maneuver_type = ManeuverType.TOPOLOGICAL_MERGE
    elif topology.source_exits_to_target:
        # Map topology connects source -> target, but the target does
        # not show multiple entries and it is not a roundabout: this is
        # not confidently a merge (could be an under-annotated topology
        # or a single-entry continuation the geometry gates above did
        # not catch) -- flag for manual REVIEW instead of guessing.
        return result(V2Decision.REVIEW, REVIEW_AMBIGUOUS_TOPOLOGY)
    else:
        return result(V2Decision.REJECT, REJECT_NO_AUTHORITATIVE_TYPE)

    if interaction.source_occupancy_frames < min_source_occupancy_frames:
        return result(V2Decision.REJECT, REJECT_INSUFFICIENT_SOURCE_HISTORY, maneuver_type)
    if interaction.longitudinal_progress_m < min_longitudinal_progress_m:
        return result(V2Decision.REJECT, REJECT_NO_PHYSICAL_TRANSITION, maneuver_type)
    if interaction.target_stable_frames < min_target_stable_frames:
        return result(V2Decision.REJECT, REJECT_TARGET_NOT_STABLE, maneuver_type)

    interactive = bool(interaction.relevant_vehicle_ids)
    if not interactive:
        return result(
            V2Decision.REJECT,
            REJECT_NO_INTERACTION,
            ManeuverType.NON_INTERACTIVE_MERGE,
            interactive=False,
        )

    both_gap_boundaries = (
        interaction.front_vehicle_id is not None
        and interaction.rear_vehicle_id is not None
    )
    if both_gap_boundaries and (
        not interaction.gap_order_valid_at_entry
        or interaction.gap_order_persistence_frames
        < interaction.required_gap_order_persistence_frames
    ):
        return result(
            V2Decision.REJECT,
            REJECT_GAP_ORDER_NOT_PERSISTENT,
            maneuver_type,
            interactive=True,
        )

    return result(V2Decision.ACCEPT, None, maneuver_type, interactive=True)


__all__ = [
    "DATASET_SCHEMA_V2",
    "LEGACY_DATASET_SCHEMA",
    "ManeuverType",
    "V2Decision",
    "TopologyEvidence",
    "InteractionEvidence",
    "V2MergeDiagnostic",
    "classify_v2_merge",
    "REJECT_SERIAL_CONTINUATION",
    "REJECT_LANE_CHANGE",
    "REJECT_CUT_IN_NOT_TOPOLOGY_MERGE",
    "REJECT_INTERSECTION",
    "REJECT_DIVERGE",
    "REJECT_MAP_ARTIFACT",
    "REJECT_NO_AUTHORITATIVE_TYPE",
    "REJECT_INSUFFICIENT_SOURCE_HISTORY",
    "REJECT_NO_PHYSICAL_TRANSITION",
    "REJECT_TARGET_NOT_STABLE",
    "REJECT_NO_INTERACTION",
    "REJECT_GAP_ORDER_NOT_PERSISTENT",
    "REVIEW_AMBIGUOUS_TOPOLOGY",
]
