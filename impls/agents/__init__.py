from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.representation_experiments.gcbc_factored_transfer import FactoredTransferGCBCAgent
from agents.representation_experiments.gciql_factored import FactoredGCIQLAgent
from agents.sac import SACAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    gcbc_factored_transfer=FactoredTransferGCBCAgent,
    gciql_factored=FactoredGCIQLAgent,
    sac=SACAgent,
)
