"""Report installed components without loading native libraries or changing state."""
import importlib.util,os,platform,shutil
print('Python/platform:',platform.python_version(),platform.platform())
for name in ['numpy','scipy','matplotlib','casadi','threadpoolctl']:
    print(name, 'available' if importlib.util.find_spec(name) else 'not installed')
for command in ['cmake','git','latexmk']:
    print(command,shutil.which(command) or 'not installed')
for name in ['FRS_SOLVER_DLL','IMITATION_NATIVE_DLL','IMITATION_ACADOS_ROOT','IMITATION_SAFEPDP_ROOT']:
    print(name,os.environ.get(name,'not set'))
