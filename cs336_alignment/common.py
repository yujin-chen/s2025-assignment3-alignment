#!/usr/bin/env python3
import pathlib

KOA_PATH =(pathlib.Path(__file__).resolve().parents[2]) / "koa_scratch/ece491b-assignment3"
FINETUNED_RESULT_PATH = (pathlib.Path(__file__).resolve().parent) / "result/finetuned_result"
BEFORE_FINETUNED_RESULT_PATH = (pathlib.Path(__file__).resolve().parent) / "result/before_finetuned_result"
DPO_RESULT_PATH = (pathlib.Path(__file__).resolve().parent) / "result/DPO_result"
DATA_PATH = (pathlib.Path(__file__).resolve().parents[1]) / "data"
SAMPLED_RESULT_PATH = (pathlib.Path(__file__).resolve().parent) / "result/sampled_result"

