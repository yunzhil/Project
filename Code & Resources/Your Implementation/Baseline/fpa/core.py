import time
from sage.base_app import BaseApp

if __name__ == '__main__':
    from FPA_algorithm import FPA
    from gaitphase import GaitPhase
    from const import SENSOR_FOOT, SENSOR_BACK, STATIONARY
    from const import NOT_STATIONARY, MIDDLE_STANCE, EARLY_STANCE, LATE_STANCE
else:
    from .FPA_algorithm import FPA
    from .gaitphase import GaitPhase
    from .const import SENSOR_FOOT, SENSOR_BACK, STATIONARY, NOT_STATIONARY
    from .const import MIDDLE_STANCE, EARLY_STANCE, LATE_STANCE

feedback_ID = [FEEDBACK_FOOT_MEDIAL, FEEDBACK_FOOT_LATERAL] = range(2)


class Core(BaseApp):
###########################################################
# INITIALIZE APP
###########################################################
    def __init__(self, my_sage):
        BaseApp.__init__(self, my_sage, __file__)

        # Set som
        self.start_time = time.time()

        # Set up the algorithm
        self.iteration = 0

        # Init the FPA phase
        self.my_GP = GaitPhase(datarate=self.info['datarate'])
        self.my_FPA = FPA(self.config['Is_right_foot'], self.info['datarate'], self.info['alpha'])
        self.feedback_foot_medial = 0
        self.feedback_foot_lateral = 0
        
###########################################################
# CHECK NODE CONNECTIONS
###########################################################
    def check_status(self):
        # check if the requirement if satisfied
        sensors_count = self.get_sensors_count()
        feedback_count = self.get_feedback_count()
        err_msg = ""
        if sensors_count < len(self.info['sensors']):
            err_msg += "Algorithm requires {} sensors but only {} are connected".format(
                len(self.info['sensors']), sensors_count)
        if feedback_count < len(self.info['feedback']):
            err_msg += "Algorithm require {} feedback but only {} are connected".format(
                len(self.info['feedback']), feedback_count)
        if err_msg != "":
            return False, err_msg
        return True, "Now running FPA-Tan algorithm"
        
###########################################################
# RUN APP IN LOOP
###########################################################
    def run_in_loop(self):
        data = self.my_sage.get_next_data()
        self.iteration += 1
        if self.iteration == 1:
            self.start_time = time.time()
        current_time = time.time()-self.start_time
        self.my_GP.update_gaitphase(data[SENSOR_FOOT])
        self.my_FPA.update_FPA(data[SENSOR_FOOT], self.my_GP.gaitphase_old, self.my_GP.gaitphase)
        FPA_M = float(self.config['FPA_medial_threshold'])
        FPA_L = float(self.config['FPA_lateral_threshold'])
        if self.my_GP.gaitphase_old == MIDDLE_STANCE and self.my_GP.gaitphase == LATE_STANCE:
            if self.my_FPA.FPA_this_step < FPA_M:
                if self.info['Is_pull']:
                    self.feedback_only_foot_L()
                else:
                    self.feedback_only_foot_M()
            elif self.my_FPA.FPA_this_step > FPA_L:
                if self.info['Is_pull']:
                    self.feedback_only_foot_M()
                else:
                    self.feedback_only_foot_L()
            else:
                self.feedback_all_off()


        my_data = {'Iteration': [self.iteration],
                   'Step_Count': [self.my_GP.step_count],
                   'Gait_Phase': [self.my_GP.gaitphase],
                   'FPA_This_Step': [self.my_FPA.FPA_this_step],
                   'FPA_Feedback_Medial': [self.feedback_foot_medial],
                   'FPA_Feedback_Lateral': [self.feedback_foot_lateral]}

        self.my_sage.save_data(data, my_data)
        self.my_sage.send_stream_data(data, my_data)

        return True

    def feedback_only_foot_M(self):
        if self.config['FPA_feedback_enabled']:
            self.my_sage.feedback_on(FEEDBACK_FOOT_MEDIAL, self.info["pulse_length"])
            self.my_sage.feedback_off(FEEDBACK_FOOT_LATERAL)
            self.feedback_foot_medial = 1
            self.feedback_foot_lateral = 0


    def feedback_only_foot_L(self):
        if self.config['FPA_feedback_enabled']:
            self.my_sage.feedback_off(FEEDBACK_FOOT_MEDIAL)
            self.my_sage.feedback_on(FEEDBACK_FOOT_LATERAL, self.info["pulse_length"])
            self.feedback_foot_medial = 0
            self.feedback_foot_lateral = 1


    def feedback_all_off(self):
        self.my_sage.feedback_off(FEEDBACK_FOOT_MEDIAL)
        self.my_sage.feedback_off(FEEDBACK_FOOT_LATERAL)
        self.feedback_foot_medial = 0
        self.feedback_foot_lateral = 0

###########################################################
# Only for testing, don't modify
###########################################################
if __name__ == '__main__':
    # This is only for testing. make sure you do the pairing first in web api
    from sage.sage import Sage

    app = Core(Sage())
    app.test_run()
