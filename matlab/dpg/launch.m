close all

basis_degree_list = [3];
tau_omega_list = [1];
n_trials = 50;
lrate_pi_list = [5e-4, 5e-3];
lrate_pi_list = [5e-4];
noise = 'uniform';

tot_count = 0;
for basis_degree = basis_degree_list
    for tau_omega = tau_omega_list
        for lrate_pi = lrate_pi_list
            tot_count = tot_count + 1;
        end
    end
end

count = 0;
for basis_degree = basis_degree_list
    for tau_omega = tau_omega_list
        for lrate_pi = lrate_pi_list
            folder_save = ['./' num2str(basis_degree) '_' num2str(tau_omega) '_' num2str(lrate_pi) '_' noise '/'];
            mkdir(folder_save)
            count = count + 1;
            args = {'basis_degree', basis_degree, 'tau_omega', tau_omega, 'lrate_pi', lrate_pi, 'mdp_noise', noise};
            for trial = 1 : n_trials
                run_dpg(trial, folder_save, args{:})
                run_dpg_notar(trial, folder_save, args{:})
                run_dpg_reg(trial, folder_save, args{:})
                run_td3(trial, folder_save, args{:})
                run_td3_reg(trial, folder_save, args{:})
                fprintf('%d / %d, %d\n', count, tot_count, trial)
            end
        end
    end
end
