from .Env_new import RealExpEnv
from .get_atom_coordinate import get_all_atom_coordinate_nm, get_atom_coordinate_nm_with_anchor
from .rrt import RRT
from .data_visualization import plot_atoms_and_design
from .assign_and_anchor import assignment, align_design, align_deisgn_stitching, get_atom_and_anchor

for .image_module import image_process

from scipy.spatial.distance import cdist as cdist
import numpy as np
from scipy.optimize import linear_sum_assignment


from collections import namedtuple
dissociate_data = namedtuple('dissociate_data',['time','x','y','current','dI_dV','topography'])


class Dissociate_Env(RealExpEnv):
    def __init__(self, step_nm, max_mvolt, max_pcurrent_to_mvolt_ratio, goal_nm, current_jump, im_size_nm, offset_nm,
                 pixel, scan_mV, max_len, safe_radius_nm = 1, speed = None, precision_lim = None):
        super(Dissociate_Env, self).__init__(step_nm, max_mvolt, max_pcurrent_to_mvolt_ratio, goal_nm, None, current_jump,
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

    def reset(self, updata_conv_net=True):
        """
        Reset the environment

        Parameters
        ----------
        update_conv_net: bool
                whether to update the parameters of the AtomJumpDetector_conv CNN

        Returns
        -------
        self.state: array_like
        info: dict
        """
        self.len = 0

#TODO  build atom_diss_detector.currents_val

        if (len(self.atom_move_detector.currents_val)>self.atom_move_detector.batch_size) and update_conv_net:
            accuracy, true_positive, true_negative = self.atom_move_detector.eval()
            self.accuracy.append(accuracy)
            self.true_positive.append(true_positive)
            self.true_negative.append(true_negative)
            self.atom_move_detector.train()

        if (self.atom_absolute_nm is None) or (self.atom_relative_nm is None):
            self.atom_absolute_nm, self.atom_relative_nm = self.scan_atom()

        if self.out_of_range(self.atom_absolute_nm, self.inner_limit_nm):
            print('Warning: atom is out of limit')
            self.pull_atom_back()
            self.atom_absolute_nm, self.atom_relative_nm = self.scan_atom()


        #goal_nm is set between 0.28 - 2 nm (Cu)
        goal_nm = self.lattice_constant + np.random.random()*(self.goal_nm - self.lattice_constant)
        print('goal_nm:',goal_nm)

        self.atom_start_absolute_nm, self.atom_start_relative_nm = self.atom_absolute_nm, self.atom_relative_nm
        # self.destination_relative_nm, self.destination_absolute_nm, self.goal = self.get_destination(self.atom_start_relative_nm, self.atom_start_absolute_nm, goal_nm)

        # self.state = np.concatenate((self.goal, (self.atom_absolute_nm - self.atom_start_absolute_nm)/self.goal_nm))
        # self.dist_destination = goal_nm
        img_forward, img_backward, offset_nm, len_nm = env.createc_controller.scan_image()

        ell_x, ell_y, ell_len, ell_wid = self.measure_fragment(img_forward)   # analyze forward or backward images
        self.state = np.array([ell_x, ell_y, ell_len, ell_wid])

        info = {'start_absolute_nm':self.atom_start_absolute_nm, 'start_relative_nm':self.atom_start_relative_nm, 'goal_absolute_nm':self.destination_absolute_nm,
                'goal_relative_nm':self.destination_relative_nm, 'start_absolute_nm_f':self.atom_absolute_nm_f, 'start_absolute_nm_b':self.atom_absolute_nm_b,
                'start_relative_nm_f':self.atom_relative_nm_f, 'start_relative_nm_b':self.atom_relative_nm_b}
        return self.state, info
    
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

        rets = self.action_to_dissociate(action)
        x_start_nm, y_start_nm, x_end_nm, y_end_nm, z_nm, mvolt, pcurrent = rets
        args = x_start_nm, y_start_nm, z_nm, mvolt
        time,V,Z,current_series, dI_dV, topography = self.step_dissociate(*args)

        info={'time':time, 'V':V, 'Z':Z, 'current_series':current_series, 'dI_dV': dI_dV, 'topography':topography, 'start_nm': np.array([x_start_nm,  y_start_nm]), 'z_nm': z_nm}

        done=False
        self.len+=1
        done = self.len==self.max_len
        if not done:
                jump = self.detect_current_jump(current_series)
        if done or jump:
                img_forward_next, img_backward_next, offset_nm, len_nm=env.createc_controller.scan_image()
                if np.abs(img_forward_next - img_forward)>1e-6 and self.measure_fragment(img_forward_next)!=self.measure_fragment(img_forward):  # if the image is obviously different from the previous one
                        done=True
# if no changes in the image or slight changes but no breakage of covalent bonds
        next_state=self.measure_fragment(img_forward_next)  # return ell_x, ell_y, ell_len, ell_wid
        reward=self.compute_reward(self.state, next_state)  # or reward=self.compute_reward(self.image_forward, image_forward_next)


        info  |= {'dist_destination':self.dist_destination,
                'atom_absolute_nm':self.atom_absolute_nm, 'atom_relative_nm':self.atom_relative_nm, 'atom_absolute_nm_f':self.atom_absolute_nm_f,
                'atom_relative_nm_f' : self.atom_relative_nm_f, 'atom_absolute_nm_b': self.atom_absolute_nm_b, 'atom_relative_nm_b':self.atom_relative_nm_b,
                'img_info':self.img_info}   #### not very sure about the img_info
        
        self.state=next_state


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
        center_x, center_y, length, width, angle: float
                the center position and size of the fragment
        """

        ell_shape=image_process(img, kernal_v=8) 


        pass ell_shape

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

    
    def step_dissociate(self, x_start_nm, y_start_nm, z_nm, mvolt, pcurrent=0.0):


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


        self.createc_controller.stm.setparam('BiasVolt.[mV]',mvolt)
        self.createc_controller.ramp_bias_mV(mvolt)
        preamp_grain = 10**float(self.createc_controller.stm.getparam("Latmangain"))
        self.createc_controller.stm.setparam("LatmanVolt",  mvolt) #(mV)
        self.createc_controller.stm.setparam("Latmanlgi", pcurrent*1e-9*preamp_grain) #(pA)
        
        self.createc_controller.set_Z_approach(z_nm)
        args = x_nm, y_nm, None, None, offset_nm, len_nm
        x_pixel, y_pixel, _, _ = self.createc_controller.nm_to_pixel(*args)
        self.createc_controller.stm.btn_tipform(x_pixel, y_pixel)
        self.createc_controller.stm.waitms(50)


        time = self.stm.vertdata(0, 0)  # time
        V= self.stm.vertdata(1,1)  # voltage
        Z = self.stm.vertdata(2,4) # Z
        current_series = self.stm.vertdata(3,3) # current series
        dI_dV = self.stm.vertdata(4,0) # dI/dV
        topography = self.stm.vertdata(15,4)
        data = dissociate_data(time,V,Z,current_series, dI_dV, topography)
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








