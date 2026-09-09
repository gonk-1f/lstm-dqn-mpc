import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from utils.final_dataset_semantics import classify_states, select_samples


POLICY = {'power_cadence_s':30., 'power_gap_limit_s':45., 'stationary_speed_kn':.1,
    'fc_noise_total_kw':4., 'battery_deadband_kw':1., 'zero_drift_kw':1.,
    'charging_min_duration_s':90., 'short_stop_max_s':900., 'terminal_settle_s':120.,
    'minimum_sailing_s':180., 'minimum_sample_s':300.}


def frame(speed, fc=100., batt=20.):
    n=len(speed)
    return pd.DataFrame({'timestamp':pd.date_range('2024-01-01',periods=n,freq='30s'),
        'parent':'p','speed_kn':speed,'fc_total_kw':fc,'battery_total_kw':batt,
        'load_total_kw':fc+batt,'aligned':True,'inverter_total_kw':30.,
        'soc_mean_pct':np.linspace(55,56,n),'soc_channel_count':12,'fc_all_stopped':fc==0})


class SemanticsTests(unittest.TestCase):
    def test_near_static_drift_does_not_restart_hours_of_dwell(self):
        d,_=classify_states(frame([0.]*5+[5.]*20+([0.]*10+[.2])*80+[0.]*5),POLICY)
        samples,_=select_samples(d,'test',POLICY)
        self.assertEqual(len(samples),1)
        self.assertLess(len(samples[0]['frame']),50)
        self.assertEqual(samples[0]['short_stop_count'],0)

    def test_impossible_measured_powers_not_hidden_by_net_cancellation(self):
        d,_=classify_states(frame([0.]*5+[5.]*20+[0.]*5,fc=1000,batt=-900),POLICY)
        self.assertTrue(d.state_class.eq('measured_power_outside_equipment_envelope').all())
        self.assertFalse(d.eligible.any())

    def test_quality_hole_inside_short_stop_does_not_end_natural_voyage(self):
        d=frame([0.]*5+[5.]*20+[0.]*5+[5.]*20+[0.]*5)
        d.loc[28,'aligned']=False
        d,_=classify_states(d,POLICY)
        samples,_=select_samples(d,'test',POLICY)
        self.assertEqual(samples,[])

    def test_near_static_drift_cannot_confirm_onboard_charging(self):
        d,_=classify_states(frame([.11]*12,fc=100,batt=-60),POLICY)
        self.assertTrue(d.state_class.eq('ambiguous_external_supply').all())

    def test_accumulated_ais_motion_is_not_sustained_sailing(self):
        d,_=classify_states(frame([0.]*5+[.2,0.]*12+[0.]*5),POLICY)
        samples,_=select_samples(d,'test',POLICY)
        self.assertEqual(samples,[])

    def test_stationary_fc_charging_not_confirmed_shore(self):
        d,t=classify_states(frame([0.]*12,fc=100,batt=-60),POLICY)
        self.assertFalse(d.state_class.eq('confirmed_shore').any())
        self.assertTrue(d.state_class.eq('ambiguous_external_supply').all())

    def test_off_stationary_sustained_charging_is_shore(self):
        d,t=classify_states(frame([0.]*12,fc=0,batt=-60),POLICY)
        self.assertTrue(d.state_class.eq('confirmed_shore').all())

    def test_short_contextual_stop_and_nonzero_service_tail_retained(self):
        speed=[0.]*5+[5.]*20+[0.]*4+[5.]*20+[0.]*5
        d,t=classify_states(frame(speed),POLICY)
        samples,exclusions=select_samples(d,'test',POLICY)
        self.assertEqual(len(samples),1)
        s=samples[0]
        self.assertTrue(s['natural_complete'])
        self.assertGreater(s['frame'].load_total_kw.iloc[-1],0)
        self.assertEqual(s['short_stop_count'],1)

    def test_quality_cut_while_sailing_is_not_a_test_voyage(self):
        d=frame([0.]*5+[5.]*30+[0.]*5)
        d.loc[20,'aligned']=False
        d,t=classify_states(d,POLICY)
        samples,exclusions=select_samples(d,'test',POLICY)
        self.assertEqual(samples,[])

    def test_train_and_validation_keep_separate_valid_blocks_beside_quality_cut(self):
        d=frame([5.]*25)
        d.loc[12,'aligned']=False
        d,_=classify_states(d,POLICY)

        train_samples,train_reasons=select_samples(d,'train',POLICY)
        validation_samples,_=select_samples(d,'validation',POLICY)
        test_samples,_=select_samples(d,'test',POLICY)

        self.assertEqual([len(sample['frame']) for sample in train_samples],[12,12])
        self.assertEqual([len(sample['frame']) for sample in validation_samples],[12,12])
        self.assertEqual(train_reasons.iloc[12],'invalid_power_alignment')
        self.assertLess(train_samples[0]['frame'].timestamp.iloc[-1],train_samples[1]['frame'].timestamp.iloc[0])
        self.assertEqual(test_samples,[])

    def test_long_true_gap_never_bridged(self):
        d=frame([0.]*5+[5.]*30+[0.]*5)
        d.loc[20:,'timestamp']+=pd.Timedelta(minutes=5)
        d,t=classify_states(d,POLICY)
        samples,_=select_samples(d,'test',POLICY)
        self.assertEqual(samples,[])


if __name__=='__main__':unittest.main()
