from .Env_new import RealExpEnv
from .get_atom_coordinate import get_all_atom_coordinate_nm, get_atom_coordinate_nm_with_anchor
from .rrt import RRT
from .data_visualization import plot_atoms_and_design
from .assign_and_anchor import assignment, align_design, align_deisgn_stitching, get_atom_and_anchor

from scipy.spatial.distance import cdist as cdist
import numpy as np
from scipy.optimize import linear_sum_assignment


from collections import namedtuple
dissociate_data = namedtuple('dissociate_data',['time','x','y','current','dI_dV','topography'])


class Dissociate_Env(RealExpEnv):
    def __init__(self, step_nm, max_mvolt, max_pcurrent_to_mvolt_ratio, goal_nm, current_jump, im_size_nm, offset_nm,
                 pixel, scan_mV, max_len, safe_radius_nm = 1, speed = None, precision_lim = None):
        super(Structure_Builder, self).__init__(step_nm, max_mvolt, max_pcurrent_to_mvolt_ratio, goal_nm, None, current_jump,
                                                im_size_nm, offset_nm, None, pixel, None, scan_mV, max_len, None, random_scan_rate = 0)
        self.atom_absolute_nm_f = None
        self.atom_absolute_nm_b = None
        self.large_offset_nm = offset_nm
        self.large_len_nm = im_size_nm
        self.safe_radius_nm = safe_radius_nm
        self.anchor_nm = None
        self.anchor_chosen = None
        if speed is None:
            self.speed = self.createc_controller.get_speed()
        else:
            self.speed = speed
        if precision_lim is not None:
            self.precision_lim = precision_lim

    def reset(self, design_nm,
                    align_design_mode = 'auto', align_design_params = {'atom_nm':None, 'design_nm':None}, sequence_mode = 'design',
                    left = None, right = None, top = None, bottom = None):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.sequence_mode = sequence_mode
        self.align_design_mode = align_design_mode
        self.num_atoms = design_nm.shape[0]
        self.all_atom_absolute_nm = self.scan_all_atoms(self.large_offset_nm, self.large_len_nm)
        if self.align_design_mode == 'auto':
            self.atoms, self.designs, anchor = align_design(self.all_atom_absolute_nm, design_nm)
            self.outside_obstacles = None
        elif self.align_design_mode =='manual':
            self.atoms, self.designs, anchor, obstacle_nm = align_deisgn_stitching(self.all_atom_absolute_nm, design_nm, align_design_params)
            if obstacle_nm is not None:
                self.outside_obstacles = list(obstacle_nm)
            else:
                self.outside_obstacles = None
        self.init_anchor = anchor
        plot_atoms_and_design(self.large_img_info, self.atoms,self.designs, self.init_anchor)
        self.design_nm = np.concatenate((self.designs, anchor.reshape((-1,2))))
        self.large_img_info |= {'design': self.design_nm}
        self.anchors = [self.init_anchor]
        offset_nm, len_nm = self.get_the_returns()
        return self.atom_chosen, self.design_chosen, self.next_destinatio_nm, self.paths, self.anchor_chosen, offset_nm, len_nm

    def step(self, action):
        """
        Take a large STM scan and update the atoms and designs after a RL episode  

        Parameters
        ----------
        succeed: bool
                if the RL episode was successful
        
        new_atom_position: array_like
                the new position of the manipulated atom

        Returns
        -------
        self.atom_chosen, self.design_chosen, self.next_destinatio_nm, self.anchor_chosen: array_like
                the positions of the atom, design, target, and anchor to be used in the RL episode 
        
        self.paths: array_like
                the planned path between atom and design
        
        offset_nm: array_like
                offset value to use for the STM scan

        len_nm: float
                image size for the STM scan 
        
        done:bool 
        """
        img_forward, img_backward, offset_nm, len_nm=env.createc_controller.scan_image()
        img_dectect_fragments=self.detect_fragments(img_forward)
        rets = self.action_to_dissociate(action)
        x_start_nm, y_start_nm, z_height_nm, bias_mv, current_pa, x_end_nm, y_end_nm = rets
        args = x_start_nm, y_start_nm, z_height_nm, bias_mv
        current, Z, V, time = self.step_dissociate(*args)

        info={'current':current, 'Z':Z, 'V':V, 'time':time, 'start_nm': np.array([x_start_nm,  y_start_nm])}

        done=False
        self.len+=1
        done = self.len==self.max_len
        if not done:
            jump = self.detect_current_jump(current_series)
            if jump:
               img_forward_next, img_backward_next, offset_nm, len_nm=env.createc_controller.scan_image()
               img_dectect_fragments_next=self.detect_fragments(img_forward_next)
               if np.abs(img_forward_next-img_forward)>1e-6 and img_dectect_fragments_next!=img_dectect_fragments:  # if the image is obviously different from the previous one
                   done=True
# if no changes in the image or slight changes but no breakage of covalent bonds
        next_state=self.measure_fragment(img_forward_next)  # return center_x, cneter_y, length, width
        reward=self.get_reward(img_forward, img_forward_next)


        info  |= {'dist_destination':self.dist_destination,
                'atom_absolute_nm':self.atom_absolute_nm, 'atom_relative_nm':self.atom_relative_nm, 'atom_absolute_nm_f':self.atom_absolute_nm_f,
                'atom_relative_nm_f' : self.atom_relative_nm_f, 'atom_absolute_nm_b': self.atom_absolute_nm_b, 'atom_relative_nm_b':self.atom_relative_nm_b,
                'img_info':self.img_info}   #### not very sure about the img_info


        return next_state, reward, done, info
    
    def measure_fragment(self, img: np.array)->tuple:   
        """
        Measure the fragment after dissociation

        Parameters
        ----------
        img: array_like
                the STM image after dissociation

        Returns
        -------
        center_x, center_y, length, width: float
                the center position and size of the fragment
        """

        pass

    def get_reward(self, img_forward: np.array, img_forward_next: np.array)->float:
        """
        Calculate the reward after dissociation

        Parameters
        ----------
        img_forward: array_like
                the STM image before dissociation

        img_forward_next: array_like
                the STM image after dissociation

        Returns
        -------
        reward: float
                the reward for the RL agent
        """
        pass

    
    def step_dissociate(self, x_start_nm, y_start_nm, z_height_nm, bias_mv, current_pa=0.0):


        '''implement the dissociation action, and collect the data'''
        x_start_nm = x_start_nm + self.atom_absolute_nm[0]
        y_start_nm = y_start_nm + self.atom_absolute_nm[1]

        x_kwargs = {'a_min':self.manip_limit_nm[0], 'a_max':self.manip_limit_nm[1]}
        y_kwargs = {'a_min':self.manip_limit_nm[2], 'a_max':self.manip_limit_nm[3]}

        x_start_nm = np.clip(x_start_nm, **x_kwargs)
        y_start_nm = np.clip(y_start_nm, **y_kwargs) 

        offset_nm=self.createc_controller.offset_nm
        len_nm=self.createc_controller.im_size_nm 

        tip_pos=x_start_nm, y_start_nm
        params = bias_mv, current_pa, offset_nm, len_nm


        self.createc_controller.stm.setparam('BiasVolt.[mV]',bias_mv)
        self.createc_controller.ramp_bias_mV(bias_mv)
        preamp_grain = 10**float(self.createc_controller.stm.getparam("Latmangain"))
        self.createc_controller.stm.setparam("LatmanVolt",  voltage) #(mV)
        self.createc_controller.stm.setparam("Latmanlgi", pcurrent*1e-9*preamp_grain) #(pA)
        
        self.createc_controller.set_Z_approach(z_height_nm)
        args = x_nm, y_nm, None, None, offset_nm, len_nm
        x_pixel, y_pixel, _, _ = self.createc_controller.nm_to_pixel(*args)
        self.createc_controller.stm.btn_tipform(x_pixel, y_pixel)
        self.createc_controller.stm.waitms(50)


        time = self.stm.vertdata(0, 0)  # time
        V= self.stm.vertdata(1,1)  # voltage
        Z = self.stm.vertdata(2,4) # Z
        current = self.stm.vertdata(3,3) # current
        dI_dV = self.stm.vertdata(4,0) # dI/dV
        topography = self.stm.vertdata(15,4)
        data = dissociate_data(time,V,Z,current, dI_dV, topography)
        # if data is not None:
        #     time = np.array(data.time).flatten()
        #     current = np.array(data.current).flatten()
        #     Z= np.array(data.Z).flatten()
        #     V= np.array(data.V).flatten()
        # else:
        #     time = None
        #     current = None
        #     Z = None
        #     V = None
        # return current, Z, V, time
        return data
    

    def detect_fragments(self, img):








