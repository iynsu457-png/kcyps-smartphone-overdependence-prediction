FEATURE_LABELS = {
    "SP_MEAN": "SAPS mean score", "SP_SUM": "SAPS sum score",
    "SP_TOLERANCE": "SAPS tolerance", "SP_WITHDRAWAL": "SAPS withdrawal",
    "SP_VIRTUAL": "SAPS virtual-life", "SP_DAILY": "Daily smartphone use",
    "SP_SUM_BASELINE": "Baseline SAPS", "SP_SUM_DELTA": "SAPS change from baseline",
    "SP_SUM_SLOPE": "SAPS slope", "PREV_HIGH_RISK": "Prior high-risk",
    "WAVE_IDX": "Wave", "COHORT_SRC_e4": "Cohort", "YGENDER": "Sex",
    "YBRT1A": "Birth year", "YBRT1B": "Birth month", "YTWIN": "Twin",
    "P_PFBRTA1": "Father age", "P_PHOMPOP": "Household size",
    "P_PINCOME": "Household income",
}
_DOM = [("YPSY1", "Self-esteem"), ("YPSY2", "Depression"), ("YPSY3", "Social anxiety"),
        ("YPSY4", "Psychological well-being"), ("YPSY5", "Peer relationship"),
        ("YPSY6", "Self-control"), ("YPSY7", "Aggression"), ("YINT", "Smartphone use"),
        ("YTIM", "Time use"), ("YMDA", "Device access"), ("YEDU", "School engagement"),
        ("YDLQ", "Delinquency"), ("YFAM", "Family relationship"),
        ("YACT", "After-school activity"), ("YFUR", "Career orientation"),
        ("YPHY", "Physical health"), ("P_PMDA", "Parental mediation"),
        ("P_PPSY", "Parenting stress"), ("P_PFAM", "Parent-reported family"),
        ("P_PEDU", "Parent education involvement"),
        ("P_PSCHOOL", "Parent school involvement"), ("P_PWORK", "Parent work hours"),
        ("P_PPHY", "Parent health"), ("P_PFBRT", "Father info"),
        ("P_PYBRT", "Mother info"), ("P_PHOM", "Household structure"),
        ("P_PINC", "Household income")]

def group_of(code):
    if code in FEATURE_LABELS:
        return FEATURE_LABELS[code]
    for pref, lab in _DOM:
        if code.startswith(pref):
            return lab
    return None
