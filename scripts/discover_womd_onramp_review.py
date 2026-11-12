#!/usr/bin/env python3
"""Bounded, local-only candidate discovery; produces review evidence, never GT.

No classifier, reward, dataset or checkpoint is changed. All metric thresholds
below rank review candidates only. Real map lines are drawn without inventing
lane polygons. The nominal-width footprint proxy is explicitly UNVERIFIED.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.render_womd_gt_merge_rollouts import load_scenario, lane_xy, track_arrays

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/womd_gt_merge_audit_20261010'
CONTROLS = {'a650cfdc29e7cc3f', 'c6201759ab89810d'}
KINDS = ('POTENTIAL_ON_RAMP', 'POTENTIAL_LANE_CHANGE', 'POTENTIAL_ZIPPER',
         'LIKELY_NON_MERGE', 'UNCERTAIN')


def csvrows(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def project(xy, line):
    """Finite segment projection with signed perpendicular d and endpoint flag.

    Unlike distance to a clamped endpoint, d never includes longitudinal travel.
    outside=True means longitudinal extrapolation would be needed; it cannot
    count as observed lane occupancy.
    """
    xy = np.atleast_2d(xy)
    delta = np.diff(line, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    keep = lengths > 1e-6
    start, delta, lengths = line[:-1][keep], delta[keep], lengths[keep]
    if not len(lengths):
        raise ValueError('degenerate lane')
    raw = np.einsum('nki,ki->nk', xy[:, None] - start, delta) / lengths**2
    feet = start + np.clip(raw, 0, 1)[..., None] * delta
    residual = xy[:, None] - feet
    idx = np.argmin(np.sum(residual**2, axis=2), axis=1)
    row = np.arange(len(xy))
    tangent = delta[idx] / lengths[idx, None]
    res = residual[row, idx]
    s = np.r_[0., np.cumsum(lengths)][idx] + np.clip(raw[row, idx], 0, 1) * lengths[idx]
    d = tangent[:, 0] * res[:, 1] - tangent[:, 1] * res[:, 0]
    outside = ((idx == 0) & (raw[row, idx] < 0)) | ((idx == len(lengths)-1) & (raw[row, idx] > 1))
    return s, d, np.arctan2(tangent[:, 1], tangent[:, 0]), outside


def longest(mask):
    best = count = 0
    for value in mask:
        count = count + 1 if value else 0
        best = max(best, count)
    return best


def track_stats(track, line, frames):
    xy, yaw, valid = track_arrays(track)
    s, d, heading, outside = project(xy, line)
    speed = np.array([np.hypot(t.velocity_x, t.velocity_y) for t in track.states])
    length = np.array([t.length for t in track.states])
    width = np.array([t.width for t in track.states])
    diff = (yaw - heading + np.pi) % (2*np.pi) - np.pi
    extent = abs(d) + abs(np.sin(diff))*length/2 + abs(np.cos(diff))*width/2
    # Screening proxy only; actual boundaries may disagree with nominal width.
    occupied = valid & ~outside & (extent <= 1.85) & (abs(diff) <= np.radians(15))
    window = np.zeros(len(valid), bool)
    window[frames[0]:frames[-1]+1] = True
    v = valid & window
    adjacent = valid[:-1] & valid[1:] & window[:-1] & window[1:]
    path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1)[adjacent].sum())
    median = float(np.median(speed[v])) if v.any() else 0.
    motion = 'MOVING' if path >= 2 and median >= 1 else 'STATIONARY' if path <= 1 and median <= .5 else 'UNCERTAIN'
    result = dict(object_id=int(track.id), valid_frames=int(v.sum()),
                  consecutive_valid_frames=longest(v), path_m=round(path, 3),
                  full_valid_path_m=round(float(np.linalg.norm(np.diff(xy,axis=0),axis=1)[valid[:-1]&valid[1:]].sum()),3),
                  measured_window_frames=[int(frames[0]),int(frames[-1])],
                  median_speed_mps=round(median, 3), motion=motion,
                  target_bbox_proxy_fraction=round(float(occupied[v].mean()), 3) if v.any() else 0.,
                  median_target_lateral_m=round(float(np.median(abs(d[v]))), 3) if v.any() else None,
                  target_lane_occupancy='UNVERIFIED',
                  occupancy_note='3.7m nominal-width oriented bbox proxy; endpoint extrapolation excluded')
    return result, (xy, valid, s, d, heading, occupied)


def shortlist(limit, seed_screening=None):
    locators = {r['scenario_id']: r for r in csvrows(ROOT/'data/manifests/v2/womd_scenario_proto_locator.csv')}
    audit = csvrows(AUDIT/'maneuver_audit.csv')
    available = []
    for row in audit:
        loc = locators.get(row['scenario_id'])
        if not loc or loc['proto_split'] != 'validation':
            continue
        shard = ROOT/'data/womd/scenario_proto/validation'/f"validation.tfrecord-{int(loc['proto_shard_index']):05d}-of-00150"
        if shard.exists():
            available.append(row)
    # Existing output labels are not score features. Raw legacy geometry is
    # only a cheap shortlist cue, to be replaced with raw-proto measurements.
    def score(r):
        history = min(int(r['commit_frame']), 90-int(r['commit_frame']))
        return min(float(r['lateral_displacement_m']), 12) + min(history, 25)*.2
    groups = [
        sorted([r for r in available if len(r['target_entry_lane_ids'].split(';')) >= 2], key=score, reverse=True),
        sorted([r for r in available if r['road_type']=='LANE_CHANGE'], key=score, reverse=True),
        sorted([r for r in available if r['v2_automatic_reason'] in ('serial_continuation','intersection_transition')], key=score, reverse=True),
        sorted(available, key=lambda r: r['candidate_id']),
    ]
    selected = {}
    if seed_screening:
        seed = json.loads(seed_screening.read_text())
        by_candidate = {r['candidate_id']: r for r in available}
        for item in seed['inspected']:
            r = by_candidate[item['candidate_id']]
            selected[r['scenario_id']] = r
        # Second bounded stratum: fast/long approach. The first pass's large
        # lateral-distance shortlist can overrepresent intersection turns.
        fast = sorted(available, key=lambda r: float(r['longitudinal_progress_m']) / max(int(r['commit_frame'])-int(r['decision_start_frame']), 5), reverse=True)
        for r in fast:
            if len(selected)>=limit: break
            if r['scenario_id'] not in selected and 15<=int(r['commit_frame'])<=70:
                selected[r['scenario_id']]=r
        return list(selected.values()), locators, len(available), len({r['scenario_id'] for r in available})
    for r in available:
        if r['scenario_id'] in CONTROLS:
            selected[r['scenario_id']] = r
    quotas = [int(limit*.55), int(limit*.2), int(limit*.1), limit]
    for group, quota in zip(groups, quotas):
        added = 0
        for r in group:
            if len(selected) >= limit or added >= quota:
                break
            if r['scenario_id'] not in selected:
                selected[r['scenario_id']] = r
                added += 1
    return list(selected.values()), locators, len(available), len({r['scenario_id'] for r in available})


def inspect(row, scenario, locator):
    source_id, target_id = int(row['source_lane_id']), int(row['target_lane_id'])
    source, target = lane_xy(scenario, source_id), lane_xy(scenario, target_id)
    lanes = {int(f.id): f.lane for f in scenario.map_features if f.HasField('lane')}
    ego = scenario.tracks[scenario.sdc_track_index]
    xy, _, valid = track_arrays(ego)
    commit = min(int(row['commit_frame']), len(xy)-1)
    frames = [max(0, commit-20), commit, min(len(xy)-1, commit+20)]
    stats, geometry = track_stats(ego, target, frames)
    _, _, ego_s, d, heading, occupied = geometry
    ss, sd, sh, so = project(xy, source)
    pre = np.arange(frames[0], commit)
    post = np.arange(commit, frames[-1]+1)
    pre_valid, post_valid = pre[valid[pre]], post[valid[post]]
    # Integrate motion along target local normals, only between consecutive
    # valid observations. This is a review cue, not a lateral-entry label.
    normal = np.c_[-np.sin(heading[:-1]), np.cos(heading[:-1])]
    lateral_steps = np.einsum('ij,ij->i', np.diff(xy, axis=0), normal)
    mask = valid[:-1] & valid[1:] & (np.arange(len(xy)-1)>=frames[0]) & (np.arange(len(xy)-1)<frames[-1])
    lateral_net = float(lateral_steps[mask].sum())
    lateral_abs = float(abs(lateral_steps[mask]).sum())
    occupancy_before = float(occupied[pre_valid].mean()) if len(pre_valid) else 0.
    occupancy_after = float(occupied[post_valid].mean()) if len(post_valid) else 0.
    # Heading change distinguishes strong turns from a near-parallel merge cue.
    _, yaw, _ = track_arrays(ego)
    heading_change = float(abs(np.degrees((yaw[frames[-1]]-yaw[frames[0]]+np.pi)%(2*np.pi)-np.pi)))
    ego_stats = dict(stats, ego_id=int(ego.id), valid_original_frames=int(valid.sum()),
        lateral_net_target_normal_m=round(lateral_net,3), lateral_abs_target_normal_m=round(lateral_abs,3),
        heading_change_deg=round(heading_change,2), target_proxy_before=round(occupancy_before,3),
        target_proxy_after=round(occupancy_after,3), target_proxy_post_stable_frames=longest(occupied[post]),
        source_pre_center_near_fraction=round(float(((abs(sd[pre_valid])<1.5)&~so[pre_valid]).mean()),3) if len(pre_valid) else 0.,
        source_to_target_entry='UNCERTAIN', lateral_motion_suspected=bool(abs(lateral_net)>=1 and heading_change<35))
    legacy_ids = {int(row[k]) for k in ('front_vehicle_id','rear_vehicle_id') if row[k]}
    neighbors = []
    for track in scenario.tracks:
        if track.id == ego.id or track.object_type != 1:
            continue
        state = track.states[commit]
        legacy = track.id in legacy_ids
        if not legacy and (not state.valid or np.hypot(state.center_x-xy[commit,0],state.center_y-xy[commit,1])>65):
            continue
        st, geom = track_stats(track, target, frames)
        st['legacy_matcher_candidate'] = legacy
        st['relative_s_at_commit_m'] = round(float(geom[2][commit]-ego_s[commit]),3) if state.valid and valid[commit] else None
        st['role_at_commit'] = ('FRONT_CANDIDATE' if st['relative_s_at_commit_m']>0 else 'REAR_CANDIDATE') if st['relative_s_at_commit_m'] is not None else 'UNVERIFIED'
        st['target_proxy_at_commit'] = bool(geom[5][commit])
        st['valid_before_and_after'] = bool(geom[1][pre].any() and geom[1][post].any())
        st['keyframe_observations'] = [dict(frame=f,valid=bool(track.states[f].valid),
            x=float(track.states[f].center_x),y=float(track.states[f].center_y),
            heading_rad=float(track.states[f].heading),
            speed_mps=float(np.hypot(track.states[f].velocity_x,track.states[f].velocity_y)),
            relative_s_m=float(geom[2][f]-ego_s[f]) if geom[1][f] and valid[f] else None,
            target_bbox_proxy=bool(geom[5][f])) for f in frames]
        st['review_priority'] = 20*legacy + 10*st['target_bbox_proxy_fraction'] + 2*(st['motion']=='MOVING') - (abs(st['relative_s_at_commit_m'] or 100)/100)
        neighbors.append(st)
    neighbors.sort(key=lambda x: x['review_priority'], reverse=True)
    neighbors = neighbors[:7]
    moving_roles = {n['role_at_commit'] for n in neighbors if n['motion']=='MOVING' and n['target_proxy_at_commit'] and n['target_bbox_proxy_fraction']>=.5 and n['consecutive_valid_frames']>=5}
    interaction = ('FRONT_REAR' if len(moving_roles)==2 else 'FRONT_ONLY' if 'FRONT_CANDIDATE' in moving_roles else 'REAR_ONLY' if moving_roles else 'UNCERTAIN')+'_CANDIDATE'
    sl, tl = lanes[source_id], lanes[target_id]
    adjacent = target_id in {int(n.feature_id) for n in list(sl.left_neighbors)+list(sl.right_neighbors)}
    convergence = target_id in sl.exit_lanes and len(tl.entry_lanes)>=2
    freeway = sl.type==1 or tl.type==1
    other_entries = [int(i) for i in tl.entry_lanes if i != source_id]
    reasons = [f'source_type={sl.type}; target_type={tl.type}; other_entries={other_entries}',
               f'ego local-normal displacement={lateral_net:.2f}m; heading change={heading_change:.1f}deg',
               f'target footprint proxy before/after={occupancy_before:.2f}/{occupancy_after:.2f}',
               f'moving neighbor screening={interaction}']
    if row['scenario_id'] in CONTROLS:
        kind='LIKELY_NON_MERGE'; reasons.insert(0, 'Mandatory negative control: known false neighbor match; road function still unreviewed')
    elif adjacent and not convergence:
        kind='POTENTIAL_LANE_CHANGE'
    elif convergence and freeway:
        kind='POTENTIAL_ON_RAMP'
    elif convergence and heading_change<20:
        kind='POTENTIAL_ZIPPER'
    elif heading_change>45 or row['v2_automatic_reason']=='serial_continuation':
        kind='LIKELY_NON_MERGE'
    else:
        kind='UNCERTAIN'
    score = 6*freeway + 4*convergence + 3*ego_stats['lateral_motion_suspected'] + 2*len(moving_roles) + min(commit,90-commit)/20 - heading_change/45
    if row['scenario_id'] in CONTROLS: score=-100
    return dict(scenario_id=scenario.scenario_id, candidate_id=row['candidate_id'], dataset_split='validation',
        locator=locator, source_lane_id=source_id, target_lane_id=target_id, source_lane_type=int(sl.type),
        target_lane_type=int(tl.type), source_exit_ids=list(sl.exit_lanes), target_entry_ids=list(tl.entry_lanes),
        source_left_boundaries=[int(b.boundary_feature_id) for b in sl.left_boundaries],
        source_right_boundaries=[int(b.boundary_feature_id) for b in sl.right_boundaries],
        target_left_boundaries=[int(b.boundary_feature_id) for b in tl.left_boundaries],
        target_right_boundaries=[int(b.boundary_feature_id) for b in tl.right_boundaries],
        automatic_candidate_category=kind, automatic_candidate_score=round(score,3),
        negative_control=scenario.scenario_id in CONTROLS, candidate_frames=frames,
        timestamps_seconds=[float(scenario.timestamps_seconds[i]) for i in frames], ego=ego_stats,
        neighbors=neighbors, expected_interaction_candidate=interaction, selection_reasons=reasons,
        uncertainties=['No independent ramp/mainline annotation; lane type/topology is a search cue only',
            'Nominal-width occupancy is not a lane-polygon verification',
            'Turning can create local-normal displacement; inspect road and ego panels together',
            'Legacy transition frame may differ from physical entry; only 9s original record',
            'Front/rear order and moving traffic do not establish gap interaction'],
        manual_labels=dict(road_type='UNCERTAIN',ego_merge='UNCERTAIN',interaction='UNCERTAIN',manual_confidence='LOW',review_notes=''))


def select(candidates, count):
    ranked=sorted(candidates, key=lambda x:x['automatic_candidate_score'], reverse=True)
    selected=[m for m in ranked if m['negative_control']]
    for kind, quota in zip(KINDS,(12,4,4,3,3)):
        pool=[m for m in ranked if m['automatic_candidate_category']==kind and m not in selected]
        selected.extend(pool[:min(quota,count-len(selected))])
    for m in ranked:
        if len(selected)>=count: break
        if m not in selected: selected.append(m)
    return sorted(selected, key=lambda m:(m['negative_control'],-m['automatic_candidate_score']))


def bbox(ax,state,color,alpha=1,label=None):
    c,s=np.cos(state.heading),np.sin(state.heading)
    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*np.array([state.length,state.width])/2
    corners=corners@np.array([[c,s],[-s,c]])+np.array([state.center_x,state.center_y])
    ax.add_patch(Polygon(corners,closed=True,facecolor=color,edgecolor=color,alpha=alpha,zorder=8))
    ax.arrow(state.center_x,state.center_y,3*c,3*s,color=color,head_width=1.2,length_includes_head=True,zorder=9)
    if label: ax.annotate(label,(state.center_x,state.center_y),xytext=(4,5),textcoords='offset points',fontsize=7,color=color,zorder=12,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.5))


def render(scenario,m,path):
    source,target=lane_xy(scenario,m['source_lane_id']),lane_xy(scenario,m['target_lane_id'])
    ego=scenario.tracks[scenario.sdc_track_index]
    ex,_,ev=track_arrays(ego)
    # Local translated coordinates are NOT used: every panel shows original x/y.
    pts=np.vstack([ex[ev],source,target])
    lo,hi=pts.min(axis=0)-15,pts.max(axis=0)+15
    center=(lo+hi)/2; span=max(hi-lo)
    limits=(center[0]-span/2,center[0]+span/2,center[1]-span/2,center[1]+span/2)
    m['plot_bounds_original_xy_m']=list(map(float,limits))
    feature_count=Counter()
    related=set(m['source_exit_ids']+m['target_entry_ids']+[m['source_lane_id'],m['target_lane_id']])
    def base(ax,labels=False):
        for f in scenario.map_features:
            kind=f.WhichOneof('feature_data')
            if kind not in ('lane','road_edge','road_line'): continue
            data=getattr(f,kind)
            xy=np.array([(p.x,p.y) for p in data.polyline])
            if len(xy)<2: continue
            if not np.any((xy[:,0]>=limits[0])&(xy[:,0]<=limits[1])&(xy[:,1]>=limits[2])&(xy[:,1]<=limits[3])): continue
            color,lw,style=('#acb7c4',.65,'-') if kind=='lane' else ('#525b64',1.1,'-') if kind=='road_edge' else ('#a9a39a',.8,'--')
            ax.plot(xy[:,0],xy[:,1],color=color,lw=lw,ls=style,alpha=.75,zorder=1)
            if labels:
                feature_count[kind]+=1
                if kind=='lane' and (int(f.id) in related or np.linalg.norm(xy[len(xy)//2]-ex[m['candidate_frames'][1]])<35):
                    mid=len(xy)//2; step=xy[min(mid+2,len(xy)-1)]-xy[max(0,mid-2)]
                    step=step/max(np.linalg.norm(step),1e-6)*4
                    ax.arrow(*xy[mid],*step,color='#64748b',head_width=1.5,length_includes_head=True)
                    ax.text(*xy[mid],str(f.id),fontsize=7,color='#475569')
        for xy,c,label in ((source,'#1565c0',f"Source {m['source_lane_id']}"),(target,'#ef8a00',f"Target {m['target_lane_id']}")):
            ax.plot(xy[:,0],xy[:,1],color=c,lw=2.5,label=label,zorder=3)
            mid=len(xy)//2; dv=xy[min(mid+2,len(xy)-1)]-xy[max(0,mid-2)]
            dv=dv/max(np.linalg.norm(dv),1e-6)*5
            ax.arrow(*xy[mid],*dv,color=c,head_width=2,length_includes_head=True,zorder=4)
        ax.set(xlim=limits[:2],ylim=limits[2:],xlabel='WOMD x [m]',ylabel='WOMD y [m]')
        ax.set_aspect('equal'); ax.ticklabel_format(style='plain',useOffset=False); ax.tick_params(labelsize=7); ax.grid(alpha=.12)
    fig=plt.figure(figsize=(22,18),layout='constrained')
    gs=fig.add_gridspec(2,2)
    axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0])]
    ax=axes[0];base(ax,True)
    ax.scatter(*target[0],marker='*',s=190,color='#ef8a00',edgecolor='black',zorder=6,label='Target start / convergence cue (unverified)')
    first=int(np.flatnonzero(ev)[0]); bbox(ax,ego.states[first],'#c62828',label=f'Ego {ego.id} start')
    ax.set_title('1. Road topology | arrows = lane travel direction',fontsize=12)
    ax.legend(loc='upper left',fontsize=8)
    ax.text(.01,.01,f"source exits={m['source_exit_ids']}\ntarget entries={m['target_entry_ids']}\nsolid dark: road edge; dashed gray: road line\nLane types: source={m['source_lane_type']}, target={m['target_lane_type']} (1=freeway, 2=surface)",transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    ax=axes[1];base(ax)
    ax.plot(np.where(ev,ex[:,0],np.nan),np.where(ev,ex[:,1],np.nan),color='#c62828',lw=2)
    points=ax.scatter(ex[ev,0],ex[ev,1],c=np.flatnonzero(ev)*.1,cmap='plasma',s=12,zorder=5)
    fig.colorbar(points,ax=ax,shrink=.4,label='Original frame time [s]')
    for label,frame in zip(('before','candidate','after'),m['candidate_frames']):
        if ev[frame]: bbox(ax,ego.states[frame],'#c62828',.6,f'{label} f{frame}')
    ax.set_title('2. Ego trajectory | candidate times, NOT confirmed merge phases',fontsize=12)
    ax.text(.01,.01,f"Ego {ego.id}; window path={m['ego']['path_m']:.1f} m\nnet local-normal motion={m['ego']['lateral_net_target_normal_m']:.2f} m\nheading change={m['ego']['heading_change_deg']:.1f} deg\nTarget bbox proxy pre/post={m['ego']['target_proxy_before']:.2f}/{m['ego']['target_proxy_after']:.2f}\nSource -> Target entry: UNCERTAIN",transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    ax=axes[2];base(ax)
    frame=m['candidate_frames'][1]
    tracks={int(t.id):t for t in scenario.tracks}
    colors=plt.get_cmap('tab10')
    palette=[0,2,4,6,7,8,9]  # Reserve red for ego and orange for target lane.
    color_by_id={n['object_id']:colors(palette[i]) for i,n in enumerate(m['neighbors'])}
    lines=[]
    for n in m['neighbors']:
        t=tracks[n['object_id']]; xy,_,v=track_arrays(t);color=color_by_id[t.id]
        ax.plot(np.where(v,xy[:,0],np.nan),np.where(v,xy[:,1],np.nan),color=color,lw=1.3)
        if v[frame]: bbox(ax,t.states[frame],color,label=str(t.id))
        lines.append(f"{t.id}: {n['role_at_commit']} {n['motion']} | {n['path_m']:.1f}m, {n['median_speed_mps']:.1f}m/s | bbox proxy={n['target_bbox_proxy_fraction']:.2f}" )
    if ev[frame]:bbox(ax,ego.states[frame],'#c62828',label='EGO')
    ax.set_title('3. Neighbor tracks | ALL Front/Rear roles and occupancy UNVERIFIED',fontsize=12)
    ax.text(.01,.01,'\n'.join(lines) or 'No selected vehicle candidates in view',transform=ax.transAxes,fontsize=7,bbox=dict(facecolor='white',alpha=.94,edgecolor='none'))
    sub=gs[1,1].subgridspec(2,2)
    for i,(label,frame) in enumerate(zip(('BEFORE','CANDIDATE','AFTER'),m['candidate_frames'])):
        ax=fig.add_subplot(sub[i//2,i%2]);base(ax)
        for t in scenario.tracks:
            if not t.states[frame].valid: continue
            st=t.states[frame]
            if not (limits[0]<=st.center_x<=limits[1] and limits[2]<=st.center_y<=limits[3]):continue
            if t.id==ego.id: bbox(ax,st,'#c62828',label=f'E {t.id}')
            elif t.id in color_by_id:bbox(ax,st,color_by_id[t.id],label=str(t.id))
            elif t.object_type==1:bbox(ax,st,'#aeb4bc',.5)
        ax.set_title(f'4.{i+1} {label}\nf{frame} / {frame*.1:.1f}s',fontsize=10)
    key=fig.add_subplot(sub[1,1]);key.axis('off')
    key.text(0,1,'4. Temporal review notes\n\nRed box: Ego\nColored boxes: UNVERIFIED neighbor IDs\nGray boxes: other observed vehicles\n\nCompare identical IDs across all 3 frames.\nFrames are legacy transition +/- 2 seconds.\nThey are NOT verified merge phases.\n\nAll map panels share original x/y limits.\nRead full resolution PNG for object IDs.\n\nCheck road function, lateral entry,\nlane occupancy, motion and observation\nlength separately before labeling.',va='top',fontsize=11,linespacing=1.6)
    m['visible_map_feature_counts']=dict(feature_count)
    fig.suptitle(f"{m['scenario_id']} | {m['automatic_candidate_category']} | score {m['automatic_candidate_score']:.2f}\nMANUAL REVIEW PENDING — NOT GROUND TRUTH"+(' | NEGATIVE MATCHER CONTROL' if m['negative_control'] else ''),fontsize=16)
    fig.savefig(path,dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/womd_onramp_positive_discovery_20261010')
    parser.add_argument('--screen-limit',type=int,default=80)
    parser.add_argument('--count',type=int,default=26)
    parser.add_argument('--seed-screening',type=Path)
    args=parser.parse_args()
    if not 20<=args.count<=30 or not args.count<=args.screen_limit<=100:parser.error('20..30 selected; bounded screening <=100')
    out=args.output_dir
    if out.exists():raise FileExistsError(f'Refusing overwrite: {out}')
    out.mkdir(parents=True)
    inputs=[AUDIT/'CORRECTION.md',AUDIT/'PHYSICAL_VALIDATION_REPORT_KO.md',AUDIT/'scenario_audit.csv',AUDIT/'maneuver_audit.csv',AUDIT/'physical_validation_sample/manual_review.csv',ROOT/'src/scenarios/physical_merge_validation.py',ROOT/'data/manifests/v2/womd_scenario_proto_locator.csv']
    fingerprints={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    rows,locators,events,scenarios=shortlist(args.screen_limit,args.seed_screening)
    candidates=[];cache={};errors=[]
    for i,row in enumerate(rows):
        try:
            scenario,_,_=load_scenario(row['scenario_id'])
            m=inspect(row,scenario,locators[row['scenario_id']]);candidates.append(m);cache[scenario.scenario_id]=scenario
        except (ValueError,KeyError,FileNotFoundError) as e:errors.append(dict(scenario_id=row['scenario_id'],error=str(e)))
        if (i+1)%10==0:print(f'Inspected {i+1}/{len(rows)} original locator records',flush=True)
    selected=select(candidates,args.count)
    if len(selected)!=args.count or not CONTROLS.issubset({m['scenario_id'] for m in selected}):raise RuntimeError('Missing candidates or required controls')
    dump(out/'screening_log.json',dict(metadata_events_available=events,metadata_scenarios_available=scenarios,records_requested=len(rows),records_inspected=len(candidates),errors=errors,seed_screening=str(args.seed_screening) if args.seed_screening else None,selection_method='Initial:55% convergence;20% lane-change;10% continuation/turn;remainder diversity. Optional extension: higher longitudinal approach speed; one event per scenario',inspected=[{k:m[k] for k in ('scenario_id','candidate_id','automatic_candidate_category','automatic_candidate_score','ego','expected_interaction_candidate')} for m in candidates]))
    for i,m in enumerate(selected,1):
        folder=out/m['scenario_id'];folder.mkdir()
        m['contact_sheet_path']=f"{m['scenario_id']}/contact_sheet.png"
        m['metadata_path']=f"{m['scenario_id']}/metadata.json"
        m['review_rank']=i
        render(cache[m['scenario_id']],m,folder/'contact_sheet.png')
        dump(folder/'metadata.json',m)
        print(f"Rendered {i}/{len(selected)} {m['scenario_id']} {m['automatic_candidate_category']}",flush=True)
    dump(out/'candidates.json',selected)
    write_csv(out/'review_template.csv',[dict(scenario_id=m['scenario_id'],road_type='UNCERTAIN',ego_merge='UNCERTAIN',interaction='UNCERTAIN',manual_confidence='LOW',review_notes='') for m in selected])
    write_csv(out/'candidate_summary.csv',[dict(scenario_id=m['scenario_id'],rank=m['review_rank'],automatic_candidate_category=m['automatic_candidate_category'],automatic_candidate_score=m['automatic_candidate_score'],negative_control=m['negative_control'],ego_lateral_motion_suspected=m['ego']['lateral_motion_suspected'],ego_lateral_net_m=m['ego']['lateral_net_target_normal_m'],moving_interaction_candidate=m['expected_interaction_candidate'],contact_sheet=m['contact_sheet_path'],metadata=m['metadata_path']) for m in selected])
    counts=Counter(m['automatic_candidate_category'] for m in selected)
    cards=[]
    for m in selected:
        sid=m['scenario_id'];esc=html.escape
        cards.append(f'<article><h2>#{m["review_rank"]} {sid}</h2><p>{m["automatic_candidate_category"]} | score {m["automatic_candidate_score"]}'+(' | NEGATIVE CONTROL' if m['negative_control'] else '')+f'</p><a href="{m["contact_sheet_path"]}"><img loading="lazy" src="{m["contact_sheet_path"]}" alt="{sid} four-panel contact sheet"></a><p><a href="{m["contact_sheet_path"]}">원본 PNG 열기</a> · <a href="{m["metadata_path"]}">Metadata JSON</a></p><p>'+esc(' / '.join(m['selection_reasons']))+'</p><p>최종 road_type / ego_merge / interaction: UNCERTAIN; 검토 대기</p></article>')
    index='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>WOMD On-Ramp 수동 검토</title><style>body{font:16px system-ui;margin:32px;background:#eef2f6;color:#18212e}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:24px}article{background:white;padding:18px;border-radius:10px}img{width:100%}h2{font-size:20px}p{line-height:1.5}a{color:#145aba}</style><h1>WOMD On-Ramp 후보 수동 검토</h1><p>자동 점수는 검토 순서입니다. 정상 On-Ramp가 확인되었다는 뜻이 아닙니다. 두 오류 사례는 Front/Rear 오분류 음성 대조군입니다.</p><p><a href="review_template.csv">사람 검토 입력 CSV</a> · <a href="candidate_summary.csv">자동 후보 요약</a> · <a href="screening_log.json">선정 이력</a> · <a href="REPORT_KO.txt">결과 보고서</a></p><details open><summary>검토 기준 / 입력값</summary><p>Source가 진입로인지, Target이 연속 본선인지, lane-change/zipper/교차로 회전인지 확인합니다. Ego의 source 주행 → 횡진입 → target 안정 주행과 관찰 길이를 확인합니다. 주변 차량의 차로 점유·이동·ID 지속성 및 실제 gap interaction을 확인합니다. 불확실하면 UNCERTAIN을 유지합니다.</p><p>road_type: ON_RAMP / LANE_CHANGE / ZIPPER / NON_MERGE / UNCERTAIN<br>ego_merge: COMPLETED / ATTEMPTED / NO_MERGE / UNCERTAIN<br>interaction: FRONT_REAR / FRONT_ONLY / REAR_ONLY / NO_NEIGHBOR / UNCERTAIN<br>manual_confidence: HIGH / MEDIUM / LOW; review_notes: 근거와 PPO 후보 적합성 및 부족한 기록을 입력</p><p>Target bbox proxy는 nominal 3.7 m 차로 폭 가정으로 계산한 탐색 수치입니다. 원본 road lines/edges와 비교해 수동 확인해야 합니다. 움직임/순서가 관찰되어도 확정 Front/Rear가 아닙니다. 모든 패널은 원본 x/y 좌표와 동일 범위입니다. Before/Candidate/After는 기존 transition frame ±2초이며 실제 합류 단계는 미확인입니다.</p></details><main>'''+''.join(cards)+'</main></html>'
    (out/'index.html').write_text(index,encoding='utf-8')
    lateral=[m['scenario_id'] for m in selected if m['ego']['lateral_motion_suspected']]
    moving=[m['scenario_id']+': '+m['expected_interaction_candidate'] for m in selected if m['expected_interaction_candidate']!='UNCERTAIN_CANDIDATE']
    report=f'''WOMD On-Ramp Positive Sample Discovery — 2026-10-10

로컬 Validation의 기존 locator {scenarios}개 / event 메타데이터 {events}건 중 제한된 {len(candidates)}개 시나리오의 원본을 확인했다. WOMD 전체 재탐색은 하지 않았다. 연결 수렴/횡이동/관찰 길이/움직이는 차량을 점수에 사용하고, lane-change·직렬 연결·교차로성 후보를 함께 포함했다. 기존 accept/MERGE_COMPLETED/FRONT_REAR를 정답으로 사용하지 않았다.

선정 및 Contact Sheet: {len(selected)}개 (음성 대조군 2개 포함).
잠정 도로 유형: {dict(counts)}
Ego 횡이동 의심 후보 (1m 이상 target-local-normal 순이동, heading 변화 35도 미만; 합류 확정 아님): {', '.join(lateral) or '없음'}
움직이는 목표 차로 차량 가능성이 있는 Front/Rear 후보 (차로 폭 proxy, 미검증): {'; '.join(moving) or '없음'}

우선 검토 Top 8:
'''+ '\n'.join(f"{i+1}. {m['scenario_id']} — {m['automatic_candidate_category']} / {m['expected_interaction_candidate']} / score={m['automatic_candidate_score']}" for i,m in enumerate(selected[:8]))+'''

index.html에서 PNG/Metadata를 열고 review_template.csv에 사람이 레이블을 입력한다. 자동 점수·분류는 candidate_summary.csv로 분리했다. 모든 사람 레이블은 UNCERTAIN, confidence LOW로 초기화했다. 이것은 사람이 LOW로 검토했다는 뜻이 아니라 검토 전 기본값이다.

한계: 9초 원본 기록, 독립 진입로/본선 annotation 부재, 일부 짧은 lane segment, 미검증 차로 폭·bbox 점유. 차로 중심선 endpoint까지 거리는 횡이동으로 사용하지 않았다. Road Edge/Line은 원본에 있는 것만 표시했다. lane type=freeway 및 여러 predecessor는 탐색 근거일 뿐 on-ramp 정답이 아니다. 상위 순위는 정답 확률이나 검증된 양성 수가 아니다.

GO: 사람이 contact sheet와 원본 metadata를 수동 검토하는 단계.
NO-GO: 정상 On-Ramp 확보 선언, Dataset v3 구축, 전체 재감사 또는 재학습. 확인된 On-Ramp는 아직 0건이며 사용자 검토 후 결정한다.

새 검토 코드와 별도 산출물만 생성했다. Dataset v1/v2, classifier, reward, checkpoint, 기존 평가 결과·GIF는 수정하지 않았다. GIF 생성, 다운로드, Git commit/push는 수행하지 않았다.
'''
    (out/'REPORT_KO.txt').write_text(report,encoding='utf-8')
    after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    assert fingerprints==after, 'Input changed during discovery'
    dump(out/'provenance.json',dict(input_sha256=fingerprints,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selected_count=len(selected),screen_limit=args.screen_limit,manual_review_status='PENDING',gif_count=0))
    print(f'Complete: {out}/index.html',flush=True)


if __name__=='__main__':
    main()
