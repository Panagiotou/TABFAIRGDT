from tabfairgdt.method.base import Method
from tabfairgdt.method.helpers import proper, smooth
from tabfairgdt.method.empty import EmptyMethod
from tabfairgdt.method.sample import SampleMethod
from tabfairgdt.method.tabfairgdt_fair_splitting_criterion import FairCART
from tabfairgdt.method.cart import CART
from tabfairgdt.method.tabfairgdt import FairCARTLeafRelabLamda


EMPTY_METHOD = ''
SAMPLE_METHOD = 'sample'
# non-parametric methods
NORMAL_CART = 'cart'

CART_FAIR_SPLITTING_METHOD = 'cart_fair_splitting'
CART_LEAF_RELAB_LAMDA = 'cart_leaf_relab_lamda'




METHODS_MAP = {EMPTY_METHOD: EmptyMethod,
               SAMPLE_METHOD: SampleMethod,
               NORMAL_CART: CART,
               CART_FAIR_SPLITTING_METHOD: FairCART,
               CART_LEAF_RELAB_LAMDA: FairCARTLeafRelabLamda,
               }


ALL_METHODS = (EMPTY_METHOD, SAMPLE_METHOD, CART_FAIR_SPLITTING_METHOD, NORMAL_CART, CART_LEAF_RELAB_LAMDA)
DEFAULT_METHODS = (CART_FAIR_SPLITTING_METHOD, NORMAL_CART, CART_LEAF_RELAB_LAMDA)
INIT_METHODS = (SAMPLE_METHOD)
NA_METHODS = (SAMPLE_METHOD)



CART_FAIR_SPLITTING_METHOD_MAP = {'int': CART_FAIR_SPLITTING_METHOD,
                   'float': CART_FAIR_SPLITTING_METHOD,
                   'datetime': CART_FAIR_SPLITTING_METHOD,
                   'bool': CART_FAIR_SPLITTING_METHOD,
                   'category': CART_FAIR_SPLITTING_METHOD
                   }
CART_LEAF_RELAB_LAMDA_MAP = {'int': CART_LEAF_RELAB_LAMDA,
                   'float': CART_LEAF_RELAB_LAMDA,
                   'datetime': CART_LEAF_RELAB_LAMDA,
                   'bool': CART_LEAF_RELAB_LAMDA,
                   'category': CART_LEAF_RELAB_LAMDA
                   }
NORMAL_CART_MAP = {'int': NORMAL_CART,
                   'float': NORMAL_CART,
                   'datetime': NORMAL_CART,
                   'bool': NORMAL_CART,
                   'category': NORMAL_CART
                   }

SAMPLE_METHOD_MAP = {'int': SAMPLE_METHOD,
                     'float': SAMPLE_METHOD,
                     'datetime': SAMPLE_METHOD,
                     'bool': SAMPLE_METHOD,
                     'category': SAMPLE_METHOD
                     }

DEFAULT_METHODS_MAP = {CART_FAIR_SPLITTING_METHOD: CART_FAIR_SPLITTING_METHOD_MAP,
                       NORMAL_CART: NORMAL_CART_MAP,
                       CART_LEAF_RELAB_LAMDA: CART_LEAF_RELAB_LAMDA_MAP
                       }


INIT_METHODS_MAP = DEFAULT_METHODS_MAP.copy()
INIT_METHODS_MAP[SAMPLE_METHOD] = SAMPLE_METHOD_MAP


CONT_TO_CAT_METHODS_MAP = {SAMPLE_METHOD: SAMPLE_METHOD,
                           CART_FAIR_SPLITTING_METHOD: CART_FAIR_SPLITTING_METHOD_MAP,
                           NORMAL_CART: NORMAL_CART,
                           CART_LEAF_RELAB_LAMDA: CART_LEAF_RELAB_LAMDA,
                           }