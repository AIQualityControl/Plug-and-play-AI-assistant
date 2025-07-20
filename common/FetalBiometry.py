#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@description:
@date       : 2022/03/25 12:20:36
@author     : Guanghua Tan
@email      : guanghuatan@hnu.edu.cn
@version    : 1.0
'''

import datetime


class FetalBiometry:

    m_idx = 0

    ac_ga = [0, 0]
    fl_ga = [0, 0]
    hc_ga = [0, 0]
    bpd_ga = [0, 0]
    hl_ga = [0, 0]
    tcd_ga = [0, 0]
    lvw_ga = [0, 0]
    pl_ga = [0, 0]
    nt_ga = [0, 0]
    crl_ga = [0, 0]
    afi_ga = [0, 0]
    q1_ga = [0, 0]
    q2_ga = [0, 0]
    q3_ga = [0, 0]
    q4_ga = [0, 0]
    ofd_ga = [0, 0]
    composite_ga_ga = [0, 0]
    spectrum_vmax_ga = [0, 0]
    spectrum_vmin_ga = [0, 0]
    spectrum_vmean_ga = [0, 0]
    spectrum_sd_ga = [0, 0]
    spectrum_ri_ga = [0, 0]
    spectrum_pi_ga = [0, 0]
    spectrum_hr_ga = [0, 0]

    spectrum_mca_vmax_ga = [0, 0]
    spectrum_mca_vmin_ga = [0, 0]
    spectrum_mca_vmean_ga = [0, 0]
    spectrum_mca_sd_ga = [0, 0]
    spectrum_mca_ri_ga = [0, 0]
    spectrum_mca_pi_ga = [0, 0]
    spectrum_mca_hr_ga = [0, 0]
    spectrum_mca_pi_ratio_ga = [0, 0]

    ga = 0
    efw = 0
    edd = ''

    # whether hc plane is detected
    is_hc_plane_detected = False

    def __init__(self):
        pass

    @classmethod
    def clear(cls):
        cls.ac_ga = [0, 0]
        cls.fl_ga = [0, 0]
        cls.hc_ga = [0, 0]
        cls.bpd_ga = [0, 0]
        cls.hl_ga = [0, 0]
        cls.tcd_ga = [0, 0]
        cls.lvw_ga = [0, 0]
        cls.pl_ga = [0, 0]
        cls.nt_ga = [0, 0]
        cls.crl_ga = [0, 0]
        cls.afi_ga = [0, 0]
        cls.q1_ga = [0, 0]
        cls.q2_ga = [0, 0]
        cls.q3_ga = [0, 0]
        cls.q4_ga = [0, 0]
        cls.ofd_ga = [0, 0]
        cls.composite_ga_ga = [0, 0]
        cls.spectrum_vmax_ga = [0, 0]
        cls.spectrum_vmin_ga = [0, 0]
        cls.spectrum_vmean_ga = [0, 0]
        cls.spectrum_sd_ga = [0, 0]
        cls.spectrum_ri_ga = [0, 0]
        cls.spectrum_pi_ga = [0, 0]
        cls.spectrum_hr_ga = [0, 0]

        cls.spectrum_mca_vmax_ga = [0, 0]
        cls.spectrum_mca_vmin_ga = [0, 0]
        cls.spectrum_mca_vmean_ga = [0, 0]
        cls.spectrum_mca_sd_ga = [0, 0]
        cls.spectrum_mca_ri_ga = [0, 0]
        cls.spectrum_mca_pi_ga = [0, 0]
        cls.spectrum_mca_hr_ga = [0, 0]
        cls.spectrum_mca_pi_ratio_ga = [0, 0]

        cls.ga = 0
        cls.efw = 0
        cls.edd = ''

        cls.is_hc_plane_detected = False

        cls.m_idx += 1

    @classmethod
    def getCaseIdx(cls):
        return cls.m_idx

    @classmethod
    def has_fl(cls):
        return cls.fl_ga[0] > 0

    @classmethod
    def has_ac(cls):
        return cls.ac_ga[0] > 0

    @classmethod
    def has_bpd(cls):
        return cls.bpd_ga[0] > 0

    @classmethod
    def has_hc(cls):
        return cls.hc_ga[0] > 0

    @classmethod
    def has_hl(cls):
        return cls.hl_ga[0] > 0

    @classmethod
    def has_tcd(cls):
        return cls.tcd_ga[0] > 0

    @classmethod
    def has_lvw(cls):
        return cls.lvw_ga[0] > 0

    @classmethod
    def has_pl(cls):
        return cls.pl_ga[0] > 0

    @classmethod
    def has_nt(cls):
        return cls.nt_ga[0] > 0

    @classmethod
    def has_crl(cls):
        return cls.crl_ga[0] > 0

    @classmethod
    def has_afi(cls):
        return cls.afi_ga[0] > 0

    @classmethod
    def has_q1(cls):
        return cls.q1_ga[0] > 0

    @classmethod
    def has_q2(cls):
        return cls.q2_ga[0] > 0

    @classmethod
    def has_q3(cls):
        return cls.q3_ga[0] > 0

    @classmethod
    def has_q4(cls):
        return cls.q4_ga[0] > 0

    @classmethod
    def has_ofd(cls):
        return cls.ofd_ga[0] > 0

    @classmethod
    def has_composite_ga(cls):
        return cls.composite_ga_ga[0] > 0

    @classmethod
    def has_spectrum_vmax(cls):
        return cls.spectrum_vmax_ga[0] > 0

    @classmethod
    def has_spectrum_vmin(cls):
        return cls.spectrum_vmin_ga[0] > 0

    @classmethod
    def has_spectrum_vmean(cls):
        return cls.spectrum_vmean_ga[0] > 0

    @classmethod
    def has_spectrum_sd(cls):
        return cls.spectrum_sd_ga[0] > 0

    @classmethod
    def has_spectrum_ri(cls):
        return cls.spectrum_ri_ga[0] > 0

    @classmethod
    def has_spectrum_pi(cls):
        return cls.spectrum_pi_ga[0] > 0

    @classmethod
    def has_spectrum_hr(cls):
        return cls.spectrum_hr_ga[0] > 0

    @classmethod
    def has_spectrum_mca_vmax(cls):
        return cls.spectrum_mca_vmax_ga[0] > 0

    @classmethod
    def has_spectrum_mca_vmin(cls):
        return cls.spectrum_mca_vmin_ga[0] > 0

    @classmethod
    def has_spectrum_mca_vmean(cls):
        return cls.spectrum_mca_vmean_ga[0] > 0

    @classmethod
    def has_spectrum_mca_sd(cls):
        return cls.spectrum_mca_sd_ga[0] > 0

    @classmethod
    def has_spectrum_mca_ri(cls):
        return cls.spectrum_mca_ri_ga[0] > 0

    @classmethod
    def has_spectrum_mca_pi(cls):
        return cls.spectrum_mca_pi_ga[0] > 0

    @classmethod
    def has_spectrum_mca_hr(cls):
        return cls.spectrum_mca_hr_ga[0] > 0

    @classmethod
    def has_spectrum_mca_pi_ratio(cls):
        return cls.spectrum_mca_pi_ratio_ga[0] > 0

    @classmethod
    def estimate_ga_efw(cls):
        if cls.has_all_biometry():
            cls.estimate_composite_age()
            cls.estimate_fetal_weight()
        else:
            cls.ga = 0
            cls.efw = 0

    @classmethod
    def estimate_composite_age(cls):
        cls.ga = cls.ga_from_ac_hc_fl_bpd(cls.ac_ga[0], cls.hc_ga[0], cls.fl_ga[0], cls.bpd_ga[0])
        dt = datetime.datetime.now()
        dt = dt + datetime.timedelta(280 - round(FetalBiometry.ga) * 7)
        cls.edd = dt.strftime('%Y%m%d')

    @classmethod
    def estimate_fetal_weight(cls):
        cls.efw = cls.efw_from_ac_hc_fl_bpd(cls.ac_ga[0], cls.hc_ga[0], cls.fl_ga[0], cls.bpd_ga[0])
        # cls.efw = cls.efw_from_ac_hc_fl(cls.ac_ga[0], cls.hc_ga[0], cls.fl_ga[0])

    @classmethod
    def get_composite_ga(cls):
        return cls.ga

    @classmethod
    def get_efw(cls):
        return cls.efw

    @classmethod
    def fl_hc_ratio(cls):
        return 0 if cls.hc_ga[0] == 0 else cls.fl_ga[0] / cls.hc_ga[0]

    @classmethod
    def hc_ac_ratio(cls):
        return 0 if cls.ac_ga[0] == 0 else cls.hc_ga[0] / cls.ac_ga[0]

    @classmethod
    def fl_ac_ratio(cls):
        return 0 if cls.ac_ga[0] == 0 else cls.fl_ga[0] / cls.ac_ga[0]

    @classmethod
    def fl_bpd_ratio(cls):
        return 0 if cls.bpd_ga[0] == 0 else cls.fl_ga[0] / cls.bpd_ga[0]

    @classmethod
    def has_all_biometry(cls):
        return cls.ac_ga[0] > 0 and cls.fl_ga[0] > 0 and cls.hc_ga[0] > 0 and cls.bpd_ga[0] > 0

    @classmethod
    def ga_from_ac(cls, ac):
        cls.ac_ga[0] = ac
        cls.ac_ga[1] = 8.14 + 0.753 * ac + 0.0036 * ac * ac
        return cls.ac_ga[1]

    @classmethod
    def ga_from_fl(cls, fl):
        cls.fl_ga[0] = fl
        cls.fl_ga[1] = 10.35 + 2.460 * fl + 0.170 * fl * fl
        return cls.fl_ga[1]

    @classmethod
    def ga_from_hl(cls, hl):
        cls.hl_ga[0] = hl
        cls.hl_ga[1] = 10.35 + 2.460 * hl + 0.170 * hl * hl
        return cls.hl_ga[1]

    @classmethod
    def ga_from_hc(cls, hc):
        cls.hc_ga[0] = hc
        cls.hc_ga[1] = 8.96 + 0.540 * hc + 0.0003 * hc * hc * hc
        return cls.hc_ga[1]

    @classmethod
    def ga_from_bpd(cls, bpd):
        cls.bpd_ga[0] = bpd
        cls.bpd_ga[1] = 9.54 + 1.482 * bpd + 0.1676 * bpd * bpd
        return cls.bpd_ga[1]

    @classmethod
    def ga_from_tcd(cls, tcd):
        cls.tcd_ga[0] = tcd
        return cls.tcd_ga[1]

    @classmethod
    def ga_from_lvw(cls, lvw):
        cls.lvw_ga[0] = lvw
        return cls.lvw_ga[1]

    @classmethod
    def ga_from_pl(cls, pl):
        cls.pl_ga[0] = pl
        return cls.pl_ga[1]

    @classmethod
    def ga_from_nt(cls, nt):
        cls.nt_ga[0] = nt
        return cls.nt_ga[1]

    @classmethod
    def ga_from_crl(cls, crl):
        cls.crl_ga[0] = crl
        return cls.crl_ga[1]

    @classmethod
    def ga_from_afi(cls, afi):
        cls.afi_ga[0] = afi
        return cls.afi_ga[1]

    @classmethod
    def ga_from_q1(cls, q1):
        cls.q1_ga[0] = q1
        return cls.q1_ga[1]

    @classmethod
    def ga_from_q2(cls, q2):
        cls.q2_ga[0] = q2
        return cls.q2_ga[1]

    @classmethod
    def ga_from_q3(cls, q3):
        cls.q3_ga[0] = q3
        return cls.q3_ga[1]

    @classmethod
    def ga_from_q4(cls, q4):
        cls.q4_ga[0] = q4
        return cls.q4_ga[1]

    @classmethod
    def ga_from_ofd(cls, ofd):
        cls.ofd_ga[0] = ofd
        return cls.ofd_ga[1]

    @classmethod
    def ga_from_composite_ga(cls, composite_ga):
        cls.composite_ga_ga[0] = composite_ga
        return cls.composite_ga_ga[1]

    @classmethod
    def ga_from_spectrum_vmax(cls, spectrum_vmax):
        cls.spectrum_vmax_ga[0] = spectrum_vmax
        return cls.spectrum_vmax_ga[1]

    @classmethod
    def ga_from_spectrum_vmin(cls, spectrum_vmin):
        cls.spectrum_vmin_ga[0] = spectrum_vmin
        return cls.spectrum_vmin_ga[1]

    @classmethod
    def ga_from_spectrum_vmean(cls, spectrum_vmean):
        cls.spectrum_vmean_ga[0] = spectrum_vmean
        return cls.spectrum_vmean_ga[1]

    @classmethod
    def ga_from_spectrum_sd(cls, spectrum_sd):
        cls.spectrum_sd_ga[0] = spectrum_sd
        return cls.spectrum_sd_ga[1]

    @classmethod
    def ga_from_spectrum_ri(cls, spectrum_ri):
        cls.spectrum_ri_ga[0] = spectrum_ri
        return cls.spectrum_ri_ga[1]

    @classmethod
    def ga_from_spectrum_pi(cls, spectrum_pi):
        cls.spectrum_pi_ga[0] = spectrum_pi
        return cls.spectrum_pi_ga[1]

    @classmethod
    def ga_from_spectrum_hr(cls, spectrum_hr):
        cls.spectrum_hr_ga[0] = spectrum_hr
        return cls.spectrum_hr_ga[1]

    @classmethod
    def ga_from_spectrum_mca_vmax(cls, spectrum_mca_vmax):
        cls.spectrum_mca_vmax_ga[0] = spectrum_mca_vmax
        return cls.spectrum_mca_vmax_ga[1]

    @classmethod
    def ga_from_spectrum_mca_vmin(cls, spectrum_mca_vmin):
        cls.spectrum_mca_vmin_ga[0] = spectrum_mca_vmin
        return cls.spectrum_mca_vmin_ga[1]

    @classmethod
    def ga_from_spectrum_mca_vmean(cls, spectrum_mca_vmean):
        cls.spectrum_mca_vmean_ga[0] = spectrum_mca_vmean
        return cls.spectrum_mca_vmean_ga[1]

    @classmethod
    def ga_from_spectrum_mca_sd(cls, spectrum_mca_sd):
        cls.spectrum_mca_sd_ga[0] = spectrum_mca_sd
        return cls.spectrum_mca_sd_ga[1]

    @classmethod
    def ga_from_spectrum_mca_ri(cls, spectrum_mca_ri):
        cls.spectrum_mca_ri_ga[0] = spectrum_mca_ri
        return cls.spectrum_mca_ri_ga[1]

    @classmethod
    def ga_from_spectrum_mca_pi(cls, spectrum_mca_pi):
        cls.spectrum_mca_pi_ga[0] = spectrum_mca_pi
        return cls.spectrum_mca_pi_ga[1]

    @classmethod
    def ga_from_spectrum_mca_pi_ratio(cls, spectrum_mca_pi_ratio):
        cls.spectrum_mca_pi_ratio_ga[0] = spectrum_mca_pi_ratio
        return cls.spectrum_mca_pi_ratio_ga[1]

    @classmethod
    def ga_from_spectrum_mca_hr(cls, spectrum_mca_hr):
        cls.spectrum_mca_hr_ga[0] = spectrum_mca_hr
        return cls.spectrum_mca_hr_ga[1]

    @classmethod
    def ga_from_hc_bpd(cls, hc, bpd):
        # cls.hc = hc
        # cls.bpd = bpd

        hc2 = hc * hc
        return 10.32 + 0.009 * hc2 + 1.320 * bpd + 0.00012 * hc2 * hc

    @classmethod
    def ga_from_ac_bpd(cls, ac, bpd):
        # cls.ac = ac
        # cls.bpd = bpd

        return 9.57 + 0.524 * ac + 0.1220 * bpd * bpd

    @classmethod
    def ga_from_fl_bpd(cls, fl, bpd):
        # cls.fl = fl
        # cls.bpd = bpd

        return 10.50 + 0.197 * bpd * fl + 0.9500 * fl + 0.7300 * bpd

    @classmethod
    def ga_from_ac_hc_fl_bpd(cls, ac, hc, fl, bpd):
        # cls.ac = ac
        # cls.hc = hc
        # cls.fl = fl
        # cls.bpd = bpd

        return 10.85 + 0.060 * hc * fl + 0.6700 * bpd + 0.1680 * ac

    @classmethod
    def efw_from_ac_hc_fl_bpd(cls, ac, hc, fl, bpd):
        weight = 1.3596 - 0.00386 * ac * fl + 0.0064 * hc + 0.00061 * bpd * ac + 0.0424 * ac + 0.174 * fl
        if weight >= 10:
            return 0
        efw = pow(10, weight)
        return efw

    @classmethod
    def efw_from_ac_hc_fl(cls, ac, hc, fl):
        weight = 1.326 - 0.00326 * ac * fl + 0.0107 * hc + 0.0438 * ac + 0.0158 * fl
        if weight >= 10:
            return 0
        efw = pow(10, weight)
        return efw


if __name__ == '__main__':
    biomerty = FetalBiometry()
    bio2 = FetalBiometry()

    biomerty.setAc(18.61)
    biomerty.setBpd(5.52)
    biomerty.setFl(4.15)
    biomerty.setHc(20.49)
    biomerty.estimateGaEfw()
    print(bio2.hasAllBiometry())

    bio2.setAc(5.52)

    print(biomerty.ac_ga)
    print(bio2.ac_ga)

    bio2.clear()
    print(biomerty.ac_ga)
    print(bio2.ac_ga)
    print(bio2.hasFl())

    biomerty.clear()
    print(biomerty.ac_ga)
    print(bio2.ac_ga)

    # biomerty.setAc(18.61)
    # print(biomerty.ac_ga)
    # print(bio2.ac_ga)
    print(biomerty.m_idx)
