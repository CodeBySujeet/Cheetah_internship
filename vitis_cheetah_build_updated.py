import torch
from torch import Tensor
from pytorch_nndct.apis import Inspector, torch_quantizer, dump_xmodel
import types
import cheetah 
import matplotlib.pyplot as plt
from cheetah import BPM, Drift, HorizontalCorrector, Segment, VerticalCorrector 
# import cheetah.converters  # noqa: F401
# from cheetah.accelerator import (  # noqa: F401
#     BPM,
#     Drift,
#     HorizontalCorrector,
#     VerticalCorrector,
# )
from cheetah.particles import ParticleBeam  # noqa: F401

batch_size = 2

# Possible DPUs for my Vitis AI version
# DPUCVDX8G_ISA3_C32B6=>0x603000b16011861
# DPUCVDX8G_ISA3_C64B1=>0x603000b16011812
# DPUCVDX8G_ISA3_C64B3=>0x603000b16011832
# DPUCVDX8G_ISA3_C64B5=>0x603000b16011852
# DPUCVDX8H_ISA1_F2W2_8PE=>0x501000000140fee
# DPUCVDX8H_ISA1_F2W4_4PE=>0x5010000001e082f
# DPUCVDX8H_ISA1_F2W4_6PE_aieDWC=>0x501000000160c2f
# DPUCVDX8H_ISA1_F2W4_6PE_aieMISC=>0x5010000001e082e

# TODO: select DPU for VCK190
# Open inspector and tell which DPU we plan to use
inspector = Inspector("DPUCVDX8G_ISA3_C32B6")

# TODO: create a segment like before
segment = Segment(
    [
        BPM(name="BPM1SMATCH"),
        Drift(length=1.0),  # Ensure length is scalar, not tensor([1.0])
        BPM(name="BPM6SMATCH"),
        Drift(length=1.0),  # Ensure length is scalar, not tensor([1.0])
        VerticalCorrector(length=0.3, angle=0.1, name="V7SMATCH"),  # Scalars, not tensors
        Drift(length=0.2),
        HorizontalCorrector(length=0.3, angle=0.1, name="H10SMATCH"),  # Scalars, not tensors
        Drift(length=7.0),
        HorizontalCorrector(length=0.3, angle=0.1, name="H12SMATCH"),  # Scalars, not tensors
        Drift(length=0.05),
        BPM(name="BPM13SMATCH"),
    ]
)
# Load a torch model And jittable will make the mode compatible with pytorch

# torch_model.__setattr__("propagate_type", {"x": Tensor, "edge_attr": Tensor})
# torch_model_jittable = segment.jittable()

# TODO: create some data for our model (e.g. particle beam)
# Define parameters for the ParticleBeam
num_particles = 10000
mu_x = torch.tensor(0.0)
mu_y = torch.tensor(0.0)
mu_xp = torch.tensor(0.0)
mu_yp = torch.tensor(0.0)
sigma_x = torch.tensor(175e-9)
sigma_y = torch.tensor(175e-9)
sigma_xp = torch.tensor(2e-7)
sigma_yp = torch.tensor(2e-7)
sigma_s = torch.tensor(1e-6)
sigma_p = torch.tensor(1e-6)
energy = torch.tensor(1e8)
total_charge = torch.tensor(0.0)
device = "cpu"
dtype = torch.float32

# Create the ParticleBeam
beam = ParticleBeam.from_parameters(
    num_particles=num_particles,
    # mu_x=mu_x,
    # mu_y=mu_y,
    # mu_xp=mu_xp,
    # mu_yp=mu_yp,
    # sigma_x=sigma_x,
    # sigma_y=sigma_y,
    # sigma_xp=sigma_xp,
    # sigma_yp=sigma_yp,
    # sigma_s=sigma_s,
    # sigma_p=sigma_p,
    # energy=energy,
    # total_charge=total_charge,
    device=device,
    dtype=dtype,
)

# Load some test data
# test_data = load_some_graph(beam)

# To run it we need to specify we are poor and we don't have a GPU
# device = torch.device("cpu")
# segment.test_energies = beam.energy
# print(beam.energy)
# inspector.inspect(segment, beam.particles, device=device)

# # After inspection proceed with quantization
# my_quantizer = torch_quantizer("test", segment, beam)
# my_quantizer.export_quant_config()
# my_quantizer.export_xmodel()

segment.track(beam)