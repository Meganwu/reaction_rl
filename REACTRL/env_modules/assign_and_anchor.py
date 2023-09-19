from .Env_new import RealExpEnv
from .get_atom_coordinate import get_all_atom_coordinate_nm, get_atom_coordinate_nm_with_anchor
from .rrt import RRT
from .data_visualization import plot_atoms_and_design

from scipy.spatial.distance import cdist as cdist
import numpy as np
from scipy.optimize import linear_sum_assignment

def assignment(start, goal):
    """
    Assign start to goal with the linear_sum_assignment function and setting the cost matrix to the distance between each start-goal pair

    Parameters
    ----------
    start, goal: array_like
        start and goal positions

    Returns
    -------
    np.array(start)[row_ind,:], np.array(goal)[col_ind,:], cost: array_like
            sorted start and goal positions, and their distances

    total_cost: float
            total distances
    
    row_ind, col_ind: array_like
            Indexes of the start and goal array in sorted order
    """
    cost_matrix = cdist(np.array(start)[:,:2], np.array(goal)[:,:2])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[row_ind, col_ind]
    total_cost = np.sum(cost)
    return np.array(start)[row_ind,:], np.array(goal)[col_ind,:], cost, total_cost, row_ind, col_ind

def align_design(atoms, design):
    """
    Move design positions and assign atoms to designs to minimize total manipulation distance 

    Parameters
    ----------
    atoms, design: array_like
        atom and design positions

    Returns
    -------
    atoms_assigned, design_assigned: array_like
            sorted atom and design (moved) positions
    
    anchor: array_like
            position of the atom that will be used as the anchor
    """
    assert atoms.shape == design.shape
    c_min = np.inf
    for i in range(atoms.shape[0]):
        for j in range(design.shape[0]):
            a = atoms[i,:]
            d = design[j,:]
            design_ = design+a-d
            a_index = np.delete(np.arange(atoms.shape[0]), i)
            d_index = np.delete(np.arange(design.shape[0]), j)
            a, d, _, c, _, _ = assignment(atoms[a_index,:], design_[d_index,:])
            if (c<c_min):
                c_min = c
                atoms_assigned, design_assigned = a, d
                anchor = atoms[i,:]
    return atoms_assigned, design_assigned, anchor

def align_deisgn_stitching(all_atom_absolute_nm, design_nm, align_design_params):
    """
    Shift the designs to match the atoms based on align_design_params. 
    Assign atoms to designs to minimize total manipulation distance.
    Get the obstacle list from align_design_params

    Parameters
    ----------
    all_atom_absolute_nm, design_nm: array_like
        atom and design positions

    align_design_params: dict
        {'atom_nm', 'design_nm', 'obstacle_nm'} 

    Returns
    -------
    atoms, designs: array_like
            sorted atom and design (moved) positions
    
    anchor_atom_nm: array_like
            position of the atom that will be used as the anchor
    """
    anchor_atom_nm = align_design_params['atom_nm']
    anchor_design_nm = align_design_params['design_nm']
    obstacle_nm = align_design_params['obstacle_nm']
    assert anchor_design_nm.tolist() in design_nm.tolist()
    dist = cdist(all_atom_absolute_nm, anchor_atom_nm.reshape((-1,2)))
    anchor_atom_nm = all_atom_absolute_nm[np.argmin(dist),:]
    atoms = np.delete(all_atom_absolute_nm, np.argmin(dist), axis=0)
    dist = cdist(design_nm, anchor_design_nm.reshape((-1,2)))
    designs = np.delete(design_nm, np.argmin(dist), axis=0)
    designs += (anchor_atom_nm - anchor_design_nm)
    if obstacle_nm is not None:
        obstacle_nm[:,:2] = obstacle_nm[:,:2]+(anchor_atom_nm - anchor_design_nm)
    return atoms, designs, anchor_atom_nm, obstacle_nm

def get_atom_and_anchor(all_atom_absolute_nm, anchor_nm):
    """
    Separate the positions of the anchor and the rest of the atoms 

    Parameters
    ----------
    all_atom_absolute_nm, anchor_nm: array_like
        positions of all the atoms and the anchor

    Returns
    -------
    atoms_nm, new_anchor_nm: array_like
            positions of all the atoms (except the anchor) and the anchor
    """
    new_anchor_nm, anchor_nm, _, _, row_ind, _ = assignment(all_atom_absolute_nm, anchor_nm)
    atoms_nm = np.delete(all_atom_absolute_nm, row_ind, axis=0)
    return atoms_nm, new_anchor_nm