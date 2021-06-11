
class BaseTrancost(object):

    @property
    def name(self):
        raise NotImplementedError()



class constant_trancost(BaseTrancost):

    @property
    def name(self):
        tcp = '_'.join(str(self.tcp).split('.'))
        return f'TCcont-{tcp}'
    
    def __init__(self, tc_para = 0.):
        self.tcp = tc_para

    def __call__(self, x):
        return self.tcp

class proportional_trancost(BaseTrancost):
    '''
    tc_para is thousandth.
    '''

    @property
    def name(self):
        tcp = '_'.join(str(self.tcp).split('.'))
        return f'TCprop-{tcp}'

    def __init__(self, tc_para = 3.):
        self.tcp = tc_para

    def __call__(self, x):
        return abs((self.tcp * x) / 1000)