import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from copy import *
thermal_current = 9e-12 # Thermischer Dunkelstrom
distance  = 1.205  # Abstand Gitter - Photodiode


#Auswertung der Mikroskopischen UNtersuchung

scaling_factor = 0.1e-3/2.83 # Meter pro Einheit
print scaling_factor
b = np.array([1.9,3.8,4.6])*scaling_factor
g = 20.8 *scaling_factor
print "Messwerte Mikroskop:"
print b,g
print "##########"
#Daten der Einzelspalte fitten
'Funktionen fuer Fit definieren'



def fit_1(phi, A, b ):
    return A**2*b**2*(np.sinc(b*np.sin(phi)/633e-9))**2

def fit_2(phi,A,b,g):
    return 4*np.cos(np.pi*np.sin(phi)*g/633e-9)**2*fit_1(phi, A, b)
i=0   


for filename in ['dataA','dataB','dataC']:
    displacement, current = np.loadtxt(filename,  unpack= "true")

    'In SI-Einheiten umrechnen und Dunkelstrom abziehen'

    displacement *= 1e-3
    current *= 1e-9
    intensity = current - thermal_current
    
    
    'In winkel umrechnen'
    max_index = intensity.argmax()
    max_location = displacement[max_index]
    max_intensity = intensity[max_index]
    angle = (displacement - max_location)/distance
    
    
    "Curve-Fit fuer Formel 5"  
    phi = np.linspace(angle[0]*0.9,angle[-1]*1.1,1000)   
    if i < 2:
        "Erwartungswert fuer A errechnen"
        A = np.sqrt(max_intensity/b[i]**2)
        params , cov = curve_fit(fit_1,angle ,intensity, (A,b[i]))
        theo_intensity = fit_1(phi,params[0],params[1])
        if i ==1:
            phi_single = copy(phi)
            theo_intensity_single = copy(theo_intensity)
            
    else:
        A = np.sqrt(max_intensity/(4*b[i]**2))
        params , cov = curve_fit(fit_2,angle ,intensity, (A,b[i],g))
        theo_intensity = fit_2(phi,params[0],params[1],params[2])
    plt.close()
    plt.plot(phi,theo_intensity, label ="Fit")
    print params       

    plt.plot(angle, intensity, 'x', label ="Messwerte")

    plt.ylabel(r'$I_{ph}$ [A]')   
    plt.xlabel(r'$\phi$ [rad]')   
    plt.legend()  
    plt.savefig('plot'+str(i)+'.png')
    i += 1
#Hier das Bild mit beiden Spalten

#Inensitaeten normieren
theo_intensity_single = theo_intensity_single/theo_intensity_single.max()
theo_intensity = theo_intensity/theo_intensity.max()

plt.close()
plt.plot(phi_single,theo_intensity_single, label ="Einzelspalt")
plt.plot(phi,theo_intensity, label ="Doppelspalt")
plt.legend()
plt.savefig('plot4.png')