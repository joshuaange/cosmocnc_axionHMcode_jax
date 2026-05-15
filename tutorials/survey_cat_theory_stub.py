"""
Catalogue module placeholder for theory-only `likelihood_type="binned"` runs.
Not used when `load_catalogue` is False.
"""


class cluster_catalogue_survey:
    def __init__(self, catalogue_name=None, observables=None, obs_select=None, cnc_params=None):
        self.catalogue = {}
        self.catalogue_patch = {}
