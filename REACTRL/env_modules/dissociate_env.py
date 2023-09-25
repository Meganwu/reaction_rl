from .Env_new import RealExpEnv
from .createc_control import Createc_Controller
from .get_atom_coordinate import get_all_atom_coordinate_nm, get_atom_coordinate_nm_with_anchor
from .rrt import RRT
from .data_visualization import plot_atoms_and_design
from .assign_and_anchor import assignment, align_design, align_deisgn_stitching, get_atom_and_anchor

from .image_module_ellipse import image_process

from scipy.spatial.distance import cdist as cdist
import numpy as np
from scipy.optimize import linear_sum_assignment
from .get_atom_coordinate import get_atom_coordinate_nm
import findiff
from .atom_jump_detection import AtomJumpDetector_conv
import os
from matplotlib import pyplot as plt, patches


from collections import namedtuple
dissociate_data = namedtuple('dissociate_data',['time','x','y','current','dI_dV','topography'])


class DissociateEnv:
        def __init__(self,
                step_nm,
                goal_nm,
                max_z_nm,
                max_mvolt,
                max_pcurrent_to_mvolt_ratio,
                template,
                current_jump,
                im_size_nm,
                offset_nm,
                manip_limit_nm,
                pixel,
                template_max_y,
                scan_mV,
                max_len,
                load_weight,
                pull_back_mV = None,
                pull_back_pA = None,
                random_scan_rate = 0.5,
                correct_drift = False,
                bottom = True,
                cellsize = 10, # nm
                max_radius = 150, # nm
                ):
                
                self.step_nm = step_nm
                self.goal_nm = goal_nm
                self.max_z_nm = max_z_nm
                self.max_mvolt = max_mvolt
                self.max_pcurrent_to_mvolt_ratio = max_pcurrent_to_mvolt_ratio
                self.pixel = pixel


                self.template = template
                args = im_size_nm, offset_nm, pixel, scan_mV
                self.createc_controller = Createc_Controller(*args)
                self.current_jump = current_jump
                self.manip_limit_nm = manip_limit_nm
                if self.manip_limit_nm is not None:
                        print('manipulation limit:', self.manip_limit_nm)
                        self.inner_limit_nm = self.manip_limit_nm + np.array([1,-1,1,-1])
                self.offset_nm = offset_nm
                self.len_nm = im_size_nm

                self.default_reward = -1
                self.default_reward_done = 1
                self.max_len = max_len
                self.correct_drift = correct_drift
                self.atom_absolute_nm = None
                self.atom_relative_nm = None
                self.template_max_y = template_max_y

                self.lattice_constant = 0.288
                self.precision_lim = self.lattice_constant*np.sqrt(3)/3
                self.bottom = bottom
                kwargs = {'data_len': 2048, 'load_weight': load_weight}
                self.atom_move_detector = AtomJumpDetector_conv(**kwargs)
                self.random_scan_rate = random_scan_rate
                self.accuracy, self.true_positive, self.true_negative = [], [], []
                if pull_back_mV is None:
                        self.pull_back_mV = 10
                else:
                        self.pull_back_mV = pull_back_mV

                if pull_back_pA is None:
                        self.pull_back_pA = 57000
                else:
                        self.pull_back_pA = pull_back_pA

                self.cellsize = cellsize
                self.max_radius = max_radius
                self.num_cell = int(self.max_radius/self.cellsize)
        

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
                return ell_shape 


    

        def compute_reward(self, img_forward: np.array, img_forward_next: np.array)->float:
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

        def action_to_diss_input(self, action):
                """
                Convert the action to the input for the dissociation

                Parameters
                ----------
                action: array_like 7D
                        the action from the RL agent

                Returns
                -------
                x_start_nm, y_start_nm, x_end_nm, y_end_nm, z_nm, mvolt, pcurrent: float
                        the input for the dissociation
                """
                x_start_nm = action[0]*self.step_nm
                y_start_nm = action[1]*self.step_nm
                x_end_nm = action[2]*self.goal_nm
                y_end_nm = action[3]*self.goal_nm
                z_nm = action[4]*self.max_z_nm
                mvolt = np.clip(action[4], a_min = None, a_max=1)*self.max_mvolt
                pcurrent = np.clip(action[5], a_min = None, a_max=1)*self.max_pcurrent_to_mvolt_ratio
                return x_start_nm, y_start_nm, x_end_nm, y_end_nm, z_nm, mvolt, pcurrent



    
        def step_dissociate(self, x_start_nm, y_start_nm, z_nm, mvolt, pcurrent):
                """
                Execute the action in Createc

                Parameters
                ----------
                x_start_nm, y_start_nm: float
                        start position of the tip dissociation in nm
                mvolt: float
                        bias voltage in mV
                pcurrent: float
                        current setpoint in pA

                Return
                ------
                current: array_like
                        manipulation current trace
                d: float
                        tip movement distance
                """



                x_start_nm = x_start_nm + self.atom_absolute_nm[0]
                y_start_nm = y_start_nm + self.atom_absolute_nm[1]

                x_kwargs = {'a_min':self.manip_limit_nm[0], 'a_max':self.manip_limit_nm[1]}
                y_kwargs = {'a_min':self.manip_limit_nm[2], 'a_max':self.manip_limit_nm[3]}

                x_start_nm = np.clip(x_start_nm, **x_kwargs)
                y_start_nm = np.clip(y_start_nm, **y_kwargs) 

                pos = x_start_nm, y_start_nm, z_nm
                params = mvolt, pcurrent, self.offset_nm, self.len_nm

                data = self.createc_controller.dissassmanipulation(*pos, *params)

                if data is not None:
                        current = np.array(data.current).flatten()
                else:
                        current = None

                return current
        
        def old_detect_current_jump(self, current):
                """
                Estimate if molecule has dissociated based on the gradient of the manipulation current trace

                Parameters
                ----------
                current: array_like
                        manipulation current trace

                Return
                ------
                bool
                whether the molecule has likely dissociated
                """
                if current is not None:
                        diff = findiff.FinDiff(0,1,acc=6)(current)[3:-3]
                        return np.sum(np.abs(diff)>self.current_jump*np.std(current)) > 2
                else:
                        return False
        
        def detect_current_jump_cnn(self, current):
                """
                Estimate if atom has moved based on AtomJumpDetector_conv and the gradient of the manipulation current trace

                Parameters
                ----------
                current: array_like
                        manipulation current trace

                Returns
                -------
                bool
                        whether the molecule has likely dissociated
                """
                if current is not None:
                        success, prediction = self.atom_diss_detector.predict(current)
                        old_prediction = self.old_detect_current_jump(current)
                        print('CNN prediction:',prediction,'M1 prediction:', old_prediction)
                        if success:
                                print('cnn thinks there is molecule dissociation')
                                return True
                        elif old_prediction and (np.random.random()>(self.random_scan_rate-0.3)):
                                return True
                        elif (np.random.random()>(self.random_scan_rate-0.2)) and (prediction>0.35):
                                print('Random scan')
                                return True
                        elif np.random.random()>self.random_scan_rate:
                                print('Random scan')
                                return True
                        else:
                                print('CNN and old prediction both say no dissociation')
                                return False
                else:
                        print('CNN and old prediction both say no dissociation')                    
                        return False

        def out_of_range(self, pos_nm, mani_limit_nm):
                """
                Check if the atom is out of the manipulation limit

                Parameters
                ----------
                pos: array_like
                        the position of the molcule in STM coordinates in nm

                mani_limit: array_like
                        [left, right, up, down] limit in STM coordinates in nm

                Returns
                -------
                bool
                        whether the atom is out of the manipulation limit
                """
                out = np.any((pos_nm-mani_limit_nm[[0,2]])*(pos_nm - mani_limit_nm[[1,3]])>0, axis=-1)
                return out       

        def pull_atom_back(self):
                """
                Pull atom to the center of self.manip_limit_nm with self.pull_back_mV, self.pull_back_pA
                """
                print('pulling atom back to center')
                current = self.pull_back_pA
                pos0 = self.atom_absolute_nm[0], self.atom_absolute_nm[1]
                pos1x = np.mean(self.manip_limit_nm[:2])+2*np.random.random()-1
                pos1y = np.mean(self.manip_limit_nm[2:])+2*np.random.random()-1
                params = self.pull_back_mV, current, self.offset_nm, self.len_nm
                self.createc_controller.lat_manipulation(*pos0, pos1x, pos1y, *params)     

        def debris_detection():
                """
                Detect debris after dissociation

                Returns
                -------
                bool
                        whether there is debris
                """
                pass      

        def atoms_detection():
                """
                Detect atoms after dissociation

                Returns
                -------
                bool
                        whether there are atoms
                """
                pass

        def crash_detection():
                """
                Detect crash after dissociation

                Returns
                -------
                bool
                        whether there is crash
                """
                pass                                                                   
                

    

   








