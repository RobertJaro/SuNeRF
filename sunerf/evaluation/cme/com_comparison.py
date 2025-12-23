import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    no_physics = np.load('/glade/work/rjarolim/sunerf-cme-v2/2view_no_physics_v01/evaluation/center_of_mass.npz', allow_pickle=True)
    physic = np.load('/glade/work/rjarolim/sunerf-cme-v2/variations/2view_040_100_v01/evaluation/center_of_mass.npz', allow_pickle=True)
    
    
    times = physic['times']
    times = (times - times.min())
    times = [t.total_seconds() / 3600 for t in times]
    
    center_of_mass_true = physic['center_of_mass_true']
    center_of_mass_physics = physic['center_of_mass_pred']
    center_of_mass_no_physics = no_physics['center_of_mass_pred']
    
    shock_front_true = physic['shock_front_true']
    shock_front_physics = physic['shock_front_pred']
    shock_front_no_physics = no_physics['shock_front_pred']
    
    mass_true = physic['mass_true']
    mass_physics = physic['mass_pred']
    mass_no_physics = no_physics['mass_pred']
    
    center_of_mass_diff_physics = physic['center_of_mass_diff']
    center_of_mass_diff_no_physics = no_physics['center_of_mass_diff']
    
    
    fig, axs = plt.subplots(1, 6, figsize=(15, 3))


    ax = axs[0]
    ax.plot(times, shock_front_true, '-o', label='Ground-truth', alpha=0.7)
    ax.plot(times, shock_front_physics, '-o', label='Physics', alpha=0.7)
    ax.plot(times, shock_front_no_physics, '-o', label='No Physics', alpha=0.7)
    ax.set_ylabel('Radial Distance [R$_\odot$]')
    ax.set_xlabel('Time [hours]')
    ax.set_title('Shock front Radius', va='bottom')
    ax.legend()

    ax = axs[1]
    ax.plot(times, center_of_mass_true[:, 0], '-o', label='Ground-truth', alpha=0.7)
    ax.plot(times, center_of_mass_physics[:, 0], '-o', label='Physics', alpha=0.7)
    ax.plot(times, center_of_mass_no_physics[:, 0], '-o', label='No Physics', alpha=0.7)
    ax.set_ylabel('Radial Distance [R$_\odot$]')
    ax.set_xlabel('Time [hours]')
    ax.set_title('CoM Radius', va='bottom')

    
    ax = axs[2]
    ax.plot(times, center_of_mass_true[:, 1], '-o', label='Ground-truth', alpha=0.7)
    ax.plot(times, center_of_mass_physics[:, 1], '-o', label='Physics', alpha=0.7)
    ax.plot(times, center_of_mass_no_physics[:, 1], '-o', label='No Physics', alpha=0.7)
    ax.set_ylabel(r'$\theta$ [deg]')
    ax.set_xlabel('Time [hours]')
    ax.set_title('CoM Latitude', va='bottom')
    
    ax = axs[3]
    ax.plot(times, center_of_mass_true[:, 2], '-o', label='Ground-truth', alpha=0.7)
    ax.plot(times, center_of_mass_physics[:, 2], '-o', label='Physics', alpha=0.7)
    ax.plot(times, center_of_mass_no_physics[:, 2], '-o', label='No Physics', alpha=0.7)
    ax.set_ylabel(r'$\phi$ [deg]')
    ax.set_xlabel('Time [hours]')
    ax.set_title('CoM Longitude', va='bottom')
    
    ax = axs[4]
    ax.plot(times, mass_true, '-o', label='Ground-truth', alpha=0.7)
    ax.plot(times, mass_physics, '-o', label='Physics', alpha=0.7)
    ax.plot(times, mass_no_physics, '-o', label='No Physics', alpha=0.7)
    ax.set_ylabel('Mass [Ne]')
    ax.set_xlabel('Time [hours]')
    ax.set_title('Total Mass', va='bottom')
    
    ax = axs[5]
    ax.plot(times, np.ones_like(times) * np.nan) # placeholder
    ax.plot(times, center_of_mass_diff_physics, '-o', label='Physics', alpha=0.7)
    ax.plot(times, center_of_mass_diff_no_physics, '-o', label='No Physics', alpha=0.7)
    ax.set_ylabel(r'$\Delta \vec{R}_\text{CoM}$ [R$_\odot$]')
    ax.set_xlabel('Time [hours]')
    ax.set_title('$\Delta$Center of Mass', va='bottom')
    
    fig.tight_layout()
    fig.savefig('/glade/work/rjarolim/sunerf-cme-v2/com_comparision.png', dpi=300, transparent=True)
    plt.close('all')