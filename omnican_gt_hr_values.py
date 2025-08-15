# Ground truth HR values for the test set in /nfs/turbo/coe-mni/iPadData_test_preprocess
# These values ensure consistent comparison between CVSM and OMNICAN
OMNICAN_GT_HR_VALUES = [
    70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 60.0, 80.0, 80.0, 
    80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 70.0, 80.0, 70.0, 70.0, 
    80.0, 70.0, 70.0, 70.0, 80.0, 80.0, 70.0, 70.0, 80.0, 80.0, 
    80.0, 90.0, 70.0, 70.0, 70.0, 90.0, 90.0, 90.0, 80.0, 90.0, 
    90.0, 80.0, 90.0, 80.0, 80.0, 80.0, 80.0, 90.0, 90.0, 80.0, 
    80.0, 80.0, 70.0, 80.0
]

def get_omnican_gt_hr_values():
    """Return the hardcoded GT HR values from OMNICAN"""
    return OMNICAN_GT_HR_VALUES.copy()
