# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>
"""Visualization utils for Argoverse MF scenarios."""

import io
import math
from pathlib import Path
from typing import Final, List, Optional, Sequence, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image as img
from PIL.Image import Image

from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory,
)
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat, NDArrayInt

_PlotBounds = Tuple[float, float, float, float]

# Configure constants
_OBS_DURATION_TIMESTEPS: Final[int] = 50
_PRED_DURATION_TIMESTEPS: Final[int] = 60

_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
_PLOT_BOUNDS_BUFFER_M: Final[float] = 30.0

_DRIVABLE_AREA_COLOR: Final[str] = "#7A7A7A"
_LANE_SEGMENT_COLOR: Final[str] = "#E0E0E0"

_DEFAULT_ACTOR_COLOR: Final[str] = "#D3E8EF"
_FOCAL_AGENT_COLOR: Final[str] = "#ECA25B"
_AV_COLOR: Final[str] = "#007672"
_BOUNDING_BOX_ZORDER: Final[
    int
] = 100  # Ensure actor bounding boxes are plotted on top of all map elements

_STATIC_OBJECT_TYPES: Set[ObjectType] = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}

def visualize_scenario_prediction(
    scenario: ArgoverseScenario,
    scenario_static_map: ArgoverseStaticMap,
    additional_traj: dict,
    traj_visible: dict,
    save_path: Path,
) -> None:
    """Build dynamic visualization for all tracks and the local map associated with an Argoverse scenario.

    Note: This function uses OpenCV to create a MP4 file using the MP4V codec.

    Args:
        scenario: Argoverse scenario to visualize.
        scenario_static_map: Local static map elements associated with `scenario`.
        save_path: Path where output MP4 video should be saved.
    """
    # Build each frame for the video
    plot_bounds: _PlotBounds = (0, 0, 0, 0)

    _, ax = plt.subplots(figsize = (20,20))

    # Plot static map elements and actor tracks
    _plot_static_map_elements_prediction(scenario_static_map)
    cur_plot_bounds = _plot_actor_tracks_prediction(ax, scenario, _OBS_DURATION_TIMESTEPS)
    plot_bounds = [1,1,1,1]
    if cur_plot_bounds:
        plot_bounds[0] = cur_plot_bounds[0]
        plot_bounds[1] = cur_plot_bounds[1]
        plot_bounds[2] = cur_plot_bounds[2]
        plot_bounds[3] = cur_plot_bounds[3]

    
    
    # Minimize plot margins and make axes invisible
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    gt_eval_world = additional_traj['gt']
    
    special_set = np.arange(gt_eval_world.shape[0])
    
    if traj_visible['gt']:
        gt_eval_world = additional_traj['gt']
        for k in range(gt_eval_world.shape[0]):
            if k in special_set:
                plt.plot(gt_eval_world[k,:,0],gt_eval_world[k,:,1],color = 'r',linewidth = 2,zorder = 1000)
            else:
                plt.plot(gt_eval_world[k,:,0],gt_eval_world[k,:,1],color = 'r',linewidth = 2,zorder = 1000)

    if traj_visible['gt_goal']:
        for k in range(gt_eval_world.shape[0]):
            if k in special_set:
                plt.scatter(gt_eval_world[k,-1,0],gt_eval_world[k,-1,1],color = 'r',marker= '*',s = 250, zorder = 1000)
            else:
                continue
                plt.scatter(gt_eval_world[k,-1,0],gt_eval_world[k,-1,1],color = 'r',marker= '*',s = 250, zorder = 1000)
            # plt.text(x=gt_eval_world[k,-1,0], y=gt_eval_world[k,-1,1],s=str(k),fontsize=50)
    
    if traj_visible['goal_point']:
        goal_point = additional_traj['goal_point']
        for k in range(goal_point.shape[0]):
            if k in special_set:
                plt.scatter(goal_point[k,0],goal_point[k,1],color = 'm',marker= 'd',s = 500, zorder = 1000000)
            else:
                continue
                plt.scatter(goal_point[k,0],goal_point[k,1],color = 'm',marker= 'd',s = 250, alpha = 0.1, zorder = 1000)
         
    if traj_visible['rec_traj']:
        rec_traj = additional_traj['rec_traj']
        if rec_traj.shape[0] == 2:
            color = ['dodgerblue','orange']
        else:
            color = ['dodgerblue']*rec_traj.shape[0]
        for k in range(rec_traj.shape[0]):
            if k in special_set:
                for i in range(rec_traj.shape[1]):
                    plt.plot(rec_traj[k,i,:50,0],rec_traj[k,i,:50,1],color = 'dodgerblue',linewidth = 6,alpha = 0.2, zorder = 10000)
                for i in range(rec_traj.shape[1]):
                    plt.plot(rec_traj[k,i,50:,0],rec_traj[k,i,50:,1],color = 'blue',linewidth = 6,alpha = 0.2, zorder = 10000)
            else:
                for i in range(rec_traj.shape[1]):
                    plt.plot(rec_traj[k,i,:,0],rec_traj[k,i,:,1],color = 'orange',linewidth = 6,alpha = 0.01, zorder = 1000)
                
    if traj_visible['marg_traj']:
        marg_traj = additional_traj['marg_traj']
        if k in special_set:
            for k in range(marg_traj.shape[0]):
                for i in range(marg_traj.shape[1]):
                    plt.plot(marg_traj[k,i,:,0],marg_traj[k,i,:,1],color = 'b',linewidth = 2,alpha = 0.5, zorder = 1000)
        else:
            for k in range(marg_traj.shape[0]):
                for i in range(marg_traj.shape[1]):
                    plt.plot(marg_traj[k,i,:,0],marg_traj[k,i,:,1],color = 'b',linewidth = 2,alpha = 0.5, zorder = 1000)
            
    # Set map bounds to capture focal trajectory history (with fixed buffer in all directions)
    plot_bounds[0] = np.min([np.min(gt_eval_world[...,0]),plot_bounds[0]])
    plot_bounds[1] = np.max([np.max(gt_eval_world[...,0]),plot_bounds[1]])
    plot_bounds[2] = np.min([np.min(gt_eval_world[...,1]),plot_bounds[2]])
    plot_bounds[3] = np.max([np.max(gt_eval_world[...,1]),plot_bounds[3]])
    
    d=15
    plt.xlim(
        plot_bounds[0] - 20,
        plot_bounds[1] + 15,
    )
    plt.ylim(
        plot_bounds[2] - 30,
        plot_bounds[3] + 5,
    )
    
    # plt.xlim(
    #     plot_bounds[0] - _PLOT_BOUNDS_BUFFER_M,
    #     plot_bounds[1] + _PLOT_BOUNDS_BUFFER_M,
    # )
    # plt.ylim(
    #     plot_bounds[2] - _PLOT_BOUNDS_BUFFER_M,
    #     plot_bounds[3] + _PLOT_BOUNDS_BUFFER_M,
    # )
    plt.gca().set_aspect("equal", adjustable="box")
    
    plt.savefig(save_path)
    plt.close()
    


def visualize_scenario(
    scenario: ArgoverseScenario,
    scenario_static_map: ArgoverseStaticMap,
    save_path: Path,
) -> None:
    """Build dynamic visualization for all tracks and the local map associated with an Argoverse scenario.

    Note: This function uses OpenCV to create a MP4 file using the MP4V codec.

    Args:
        scenario: Argoverse scenario to visualize.
        scenario_static_map: Local static map elements associated with `scenario`.
        save_path: Path where output MP4 video should be saved.
    """
    # Build each frame for the video
    frames: List[Image] = []
    plot_bounds: _PlotBounds = (0, 0, 0, 0)

    for timestep in range(_OBS_DURATION_TIMESTEPS + _PRED_DURATION_TIMESTEPS):
        _, ax = plt.subplots()

        # Plot static map elements and actor tracks
        _plot_static_map_elements(scenario_static_map)
        cur_plot_bounds = _plot_actor_tracks(ax, scenario, timestep)
        if cur_plot_bounds:
            plot_bounds = cur_plot_bounds

        # Set map bounds to capture focal trajectory history (with fixed buffer in all directions)
        plt.xlim(
            plot_bounds[0] - _PLOT_BOUNDS_BUFFER_M,
            plot_bounds[1] + _PLOT_BOUNDS_BUFFER_M,
        )
        plt.ylim(
            plot_bounds[2] - _PLOT_BOUNDS_BUFFER_M,
            plot_bounds[3] + _PLOT_BOUNDS_BUFFER_M,
        )
        plt.gca().set_aspect("equal", adjustable="box")

        # Minimize plot margins and make axes invisible
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # Save plotted frame to in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        frame = img.open(buf)
        frames.append(frame)

    # Write buffered frames to MP4V-encoded video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_path = str(save_path.parents[0] / f"{save_path.stem}.mp4")
    video = cv2.VideoWriter(vid_path, fourcc, fps=10, frameSize=frames[0].size)
    for i in range(len(frames)):
        frame_temp = frames[i].copy()
        video.write(cv2.cvtColor(np.array(frame_temp), cv2.COLOR_RGB2BGR))
    video.release()

def _plot_static_map_elements_prediction(
    static_map: ArgoverseStaticMap, show_ped_xings: bool = False
) -> None:
    """Plot all static map elements associated with an Argoverse scenario.

    Args:
        static_map: Static map containing elements to be plotted.
        show_ped_xings: Configures whether pedestrian crossings should be plotted.
    """
    # Plot drivable areas
    # for drivable_area in static_map.vector_drivable_areas.values():
    #     _plot_polygons([drivable_area.xyz], alpha=0.5, color=_DRIVABLE_AREA_COLOR)

    # Plot lane segments
    for lane_segment in static_map.vector_lane_segments.values():
        _plot_polylines(
            [
                lane_segment.left_lane_boundary.xyz,
                lane_segment.right_lane_boundary.xyz,
            ],
            line_width=3,
            color=_LANE_SEGMENT_COLOR,
        )

    # Plot pedestrian crossings
    if show_ped_xings:
        for ped_xing in static_map.vector_pedestrian_crossings.values():
            _plot_polylines(
                [ped_xing.edge1.xyz, ped_xing.edge2.xyz],
                alpha=1.0,
                color=_LANE_SEGMENT_COLOR,
            )

def _plot_static_map_elements(
    static_map: ArgoverseStaticMap, show_ped_xings: bool = False
) -> None:
    """Plot all static map elements associated with an Argoverse scenario.

    Args:
        static_map: Static map containing elements to be plotted.
        show_ped_xings: Configures whether pedestrian crossings should be plotted.
    """
    # Plot drivable areas
    for drivable_area in static_map.vector_drivable_areas.values():
        _plot_polygons([drivable_area.xyz], alpha=0.5, color=_DRIVABLE_AREA_COLOR)

    # Plot lane segments
    for lane_segment in static_map.vector_lane_segments.values():
        _plot_polylines(
            [
                lane_segment.left_lane_boundary.xyz,
                lane_segment.right_lane_boundary.xyz,
            ],
            line_width=0.5,
            color=_LANE_SEGMENT_COLOR,
        )

    # Plot pedestrian crossings
    if show_ped_xings:
        for ped_xing in static_map.vector_pedestrian_crossings.values():
            _plot_polylines(
                [ped_xing.edge1.xyz, ped_xing.edge2.xyz],
                alpha=1.0,
                color=_LANE_SEGMENT_COLOR,
            )

def _plot_actor_tracks_prediction(
    ax: plt.Axes, scenario: ArgoverseScenario, timestep: int
) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.

    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    for track in scenario.tracks:
        # Get timesteps for which actor data is valid
        actor_timesteps: NDArrayInt = np.array(
            [
                object_state.timestep
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )
        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            continue

        # Get actor trajectory and heading history
        actor_trajectory: NDArrayFloat = np.array(
            [
                list(object_state.position)
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )
        actor_headings: NDArrayFloat = np.array(
            [
                object_state.heading
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )

        # Plot polyline for focal agent location history
        track_color = _DEFAULT_ACTOR_COLOR
        # if track.track_id == "AV":
        #     track_color = _AV_COLOR
            
        if track.category == TrackCategory.FOCAL_TRACK or track.category == TrackCategory.SCORED_TRACK:
            x_min, x_max = actor_trajectory[:, 0].min(), actor_trajectory[:, 0].max()
            y_min, y_max = actor_trajectory[:, 1].min(), actor_trajectory[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)
            track_color = _AV_COLOR

        # Plot bounding boxes for all vehicles and cyclists
        if track.object_type == ObjectType.VEHICLE:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
            )
        elif (
            track.object_type == ObjectType.CYCLIST
            or track.object_type == ObjectType.MOTORCYCLIST
        ):
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
            )
        else:
            plt.plot(
                actor_trajectory[-1, 0],
                actor_trajectory[-1, 1],
                "o",
                color=track_color,
                markersize=4,
            )

    return track_bounds

def _plot_actor_tracks(
    ax: plt.Axes, scenario: ArgoverseScenario, timestep: int
) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.

    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    for track in scenario.tracks:
        # Get timesteps for which actor data is valid
        actor_timesteps: NDArrayInt = np.array(
            [
                object_state.timestep
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )
        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            continue

        # Get actor trajectory and heading history
        actor_trajectory: NDArrayFloat = np.array(
            [
                list(object_state.position)
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )
        actor_headings: NDArrayFloat = np.array(
            [
                object_state.heading
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )

        # Plot polyline for focal agent location history
        track_color = _DEFAULT_ACTOR_COLOR
        if track.category == TrackCategory.FOCAL_TRACK:
            x_min, x_max = actor_trajectory[:, 0].min(), actor_trajectory[:, 0].max()
            y_min, y_max = actor_trajectory[:, 1].min(), actor_trajectory[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)
            track_color = _FOCAL_AGENT_COLOR
            _plot_polylines([actor_trajectory], color=track_color, line_width=2)
        elif track.track_id == "AV":
            track_color = _AV_COLOR
        elif track.object_type in _STATIC_OBJECT_TYPES:
            continue

        # Plot bounding boxes for all vehicles and cyclists
        if track.object_type == ObjectType.VEHICLE:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
            )
        elif (
            track.object_type == ObjectType.CYCLIST
            or track.object_type == ObjectType.MOTORCYCLIST
        ):
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
            )
        else:
            plt.plot(
                actor_trajectory[-1, 0],
                actor_trajectory[-1, 1],
                "o",
                color=track_color,
                markersize=4,
            )

    return track_bounds


def _plot_polylines(
    polylines: Sequence[NDArrayFloat],
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    for polyline in polylines:
        plt.plot(
            polyline[:, 0],
            polyline[:, 1],
            style,
            linewidth=line_width,
            color=color,
            alpha=alpha,
        )


def _plot_polygons(
    polygons: Sequence[NDArrayFloat], *, alpha: float = 1.0, color: str = "r"
) -> None:
    """Plot a group of filled polygons with the specified config.

    Args:
        polygons: Collection of polygons specified by (N,2) arrays of vertices.
        alpha: Desired alpha for the polygon fill.
        color: Desired color for the polygon.
    """
    for polygon in polygons:
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)


def _plot_actor_bounding_box(
    ax: plt.Axes,
    cur_location: NDArrayFloat,
    heading: float,
    color: str,
    bbox_size: Tuple[float, float],
) -> None:
    """Plot an actor bounding box centered on the actor's current location.

    Args:
        ax: Axes on which actor bounding box should be plotted.
        cur_location: Current location of the actor (2,).
        heading: Current heading of the actor (in radians).
        color: Desired color for the bounding box.
        bbox_size: Desired size for the bounding box (length, width).
    """
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y),
        bbox_length,
        bbox_width,
        angle=np.degrees(heading),
        
        # edgecolor = 'k',
        # linewidth = 2,
        facecolor=color,
        zorder=_BOUNDING_BOX_ZORDER,
    )
    ax.add_patch(vehicle_bounding_box)