import subprocess


subprocess.run(['python', 'Sigmoid1L.py', "e", "fwhm_x", "fwhm_y", "co_max", "cx_max"])
subprocess.run(['python', 'Sigmoid2L.py', "e", "fwhm_x", "fwhm_y", "co_max", "cx_max"])
subprocess.run(['python', 'Sigmoid3L.py', "e", "fwhm_x", "fwhm_y", "co_max", "cx_max"])
subprocess.run(['python', 'Tanh1L.py', "e", "fwhm_x", "fwhm_y", "co_max", "cx_max"])
subprocess.run(['python', 'Tanh2L.py', "e", "fwhm_x", "fwhm_y", "co_max", "cx_max"])
subprocess.run(['python', 'Tanh3L.py', "e", "fwhm_x", "fwhm_y", "co_max", "cx_max"])


# RUN TEST FOR ARCHITECTURE VARIABILITY

#subprocess.run(['python', 'Test3LSig.py', "e", "fwhm_x", "fwhm_y", "co_max"])
#subprocess.run(['python', 'Test2LSig.py', "e", "fwhm_x", "fwhm_y", "co_max"])
#subprocess.run(['python', 'Test1LSig.py', "e", "fwhm_x", "fwhm_y", "co_max", "cx_max"])

#subprocess.run(['python', 'Test1LTanh.py', "e", "fwhm_x", "fwhm_y", "co_max"])
#subprocess.run(['python', 'Test2LTanh.py', "e", "fwhm_x", "fwhm_y", "co_max"])
#subprocess.run(['python', 'Test3LTanh.py', "e", "fwhm_x", "fwhm_y", "co_max", "cx_max"])

#subprocess.run(['python', 'CheckRatio_Sigmoid1L.py'])
#subprocess.run(['python', 'CheckRatio_Sigmoid2L.py'])
#subprocess.run(['python', 'CheckRatio_Sigmoid3L.py'])