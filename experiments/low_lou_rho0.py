from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nfgs.workflow import experiment_main
if __name__=='__main__':experiment_main('low_lou_rho0')
