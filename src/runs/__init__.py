REGISTRY = {}


from .run_AIQMIX import run as run_AIQMIX
REGISTRY["AIQMIX"] = run_AIQMIX

from .run_CollaQ import run as run_CollaQ
REGISTRY["CollaQ"] = run_CollaQ

from .run_NDQ import run as run_NDQ
REGISTRY["NDQ"] = run_NDQ

from .run_QPLEX import run as run_QPLEX
REGISTRY["QPLEX"] = run_QPLEX

from .run_RODE import run as run_RODE
REGISTRY["RODE"] = run_RODE

from .run_ROMA import run as run_ROMA
REGISTRY["ROMA"] = run_ROMA

from .run_LICA import run as run_LICA
REGISTRY["LICA"] = run_LICA

from .run_ASN import run as run_ASN
REGISTRY["ASN"] = run_ASN




