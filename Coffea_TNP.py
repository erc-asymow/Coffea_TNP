import awkward as ak
import hist
from coffea.processor import dict_accumulator, column_accumulator, defaultdict_accumulator
from coffea import processor
from coffea.nanoevents.methods import candidate
from functools import partial
import numba
import numpy as np
import uproot
import uproot3
from coffea.nanoevents import NanoEventsFactory, BaseSchema, NanoAODSchema
from os import listdir
from os.path import isfile,join
#import ROOT


@numba.njit
def find_TNPPair(events_leptons, builder):
    """Search for valid 4-lepton combinations from an array of events * leptons {charge, ...}
    A valid candidate has two pairs of leptons that each have balanced charge
    Outputs an array of events * candidates {indices 0..3} corresponding to all valid
    permutations of all valid combinations of unique leptons in each event
    (omitting permutations of the pairs)
    """
    for leptons in events_leptons:
        builder.begin_list()
        nlep = len(leptons)
        for i0 in range(nlep):
            if (not(leptons[i0].isGlobal and (leptons[i0].pt > 20) and (abs(leptons[i0].eta) < 2 ) )) : 
                continue
            for i1 in range(nlep):
                if leptons[i0].charge + leptons[i1].charge != 0: 
                    continue
                if (not(leptons[i1].isTracker and (leptons[i1].pt > 10) )) : 
                    continue
                builder.begin_tuple(2)
                builder.index(0).integer(i0)
                builder.index(1).integer(i1)
                builder.end_tuple()
        builder.end_list()

    return builder


class FancyDimuonProcessor(processor.ProcessorABC):
    def process(self, events):
        dataset = events.metadata['dataset']
        muons = events.Muon 
        
        # make sure they are sorted by transverse momentum
        #muons = muons[ak.argsort(muons.pt, axis=1)]
        
        # track our cuts' yields
        cutflow = {}
        cutflow['all events'] = len(muons)
        
        # impose some quality and minimum pt cuts on the muons
        #muons = muons[
        #    muons.softId
        #    & (muons.pt > 5)
        #    & (muons.isolation < 0.2)
        #]
        cutflow['at least 2 muons'] = ak.sum(ak.num(muons) >= 2)
        #nmuons = hist.Hist.new.Reg(6, 0, 6, label="N good muons").Double().fill(ak.num(muons))
        
        # reduce first axis: skip events without enough muons
        muons = muons[ak.num(muons) >= 2]
        
        # find all candidates with helper function
        TNPPair = find_TNPPair(muons, ak.ArrayBuilder()).snapshot()
        if ak.all(ak.num(TNPPair) == 0):
            # skip processing as it is an EmptyArray
            return {}

        TNPPair = [muons[TNPPair[idx]] for idx in "01"]
        TNPPair = ak.zip({
             "Tag": TNPPair[0],
             "Probe": TNPPair[1],
             "p4": TNPPair[0] + TNPPair[1],
        })
        
        cutflow['at least one candidate'] = ak.sum(ak.num(TNPPair) > 0)
         
        # require minimum dimuon mass
        TNPPair = TNPPair[(TNPPair.p4.mass > 40.) & (TNPPair.p4.mass <200.)]
        cutflow['minimum dimuon mass'] = ak.sum(ak.num(TNPPair) > 0)
        
        # choose permutation with z1 mass closest to nominal Z boson mass
        #bestz1 = ak.singletons(ak.argmin(abs(fourmuon.z1.p4.mass - 91.1876), axis=1))
        #fourmuon = ak.flatten(fourmuon[bestz1])
        
        out = {
            "cutflow": cutflow,
            #"nmuons": nmuons,
            "mass": column_accumulator(ak.to_numpy(ak.flatten(TNPPair.p4.mass))),
            "pt": column_accumulator(ak.to_numpy(ak.flatten(TNPPair.Probe.pt))),
            "eta": column_accumulator(ak.to_numpy(ak.flatten(TNPPair.Probe.eta))),
            "isGlobal": column_accumulator((ak.to_numpy(ak.flatten(TNPPair.Probe.isGlobal))).astype(np.int32)),
            "pfRelIso04_all": column_accumulator(ak.to_numpy(ak.flatten(TNPPair.Probe.pfRelIso04_all))),
        }
        return {dataset: out}

    def postprocess(self, accumulator):
        return accumulator


import time

tstart = time.time()    
cpustrat = time.process_time()

dirname="/scratchnvme/wmass/NANOJEC/postVFP/DYJetsToMuMu/"
files = [join(dirname, f) for f in listdir(dirname) if f.endswith(".root")]

print(files)

fileset = {
    'Drell_Yan': files
}

output = processor.run_uproot_job(
    fileset,
    treename='Events',
    processor_instance=FancyDimuonProcessor(),
    executor=processor.futures_executor,
    executor_args={"schema": NanoAODSchema, "workers": 128},
    #executor=processor.iterative_executor,
    #executor_args={
    #    "schema": NanoAODSchema
    #},
    #chunksize=100_000,
    #maxchunks=30,
)

print(output["Drell_Yan"]["cutflow"])

tree_start = time.time()
tree_start_cpu = time.process_time() 
outputfile = uproot.recreate("output9.root")
outputfile["tpTree/fitter_tree"] = {"mass": output["Drell_Yan"]["mass"].value, "pt": output["Drell_Yan"]["pt"].value, "eta": output["Drell_Yan"]["eta"].value, "isGlobal": output["Drell_Yan"]["isGlobal"].value, "pfRelIso04_all": output["Drell_Yan"]["pfRelIso04_all"].value}
#outputfile["TNPTree"] = output["ZZ to 4mu"]["TNPTree"]
outputfile.close()
print('Tree writing time:', time.time() - tree_start,'seconds')
print('Tree writing cpu time:', time.process_time() - tree_start_cpu,'seconds')
elapsed = time.time() - tstart
elapsed_cpu = time.process_time() - cpustrat
print('Execution time:', elapsed, 'seconds')
print('CPU Execution time:', elapsed_cpu , 'seconds')
