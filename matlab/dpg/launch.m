close all

for basis_degree = 2 : 3
    for tau_omega = [0.01, 1]
        folder_save = ['./' num2str(basis_degree) '_' num2str(tau_omega) '_noise/'];
        mkdir(folder_save)
        parfor trial = 1 : 50
%             run_dpg(trial,folder_save,'basis_degree',basis_degree,'tau_omega',tau_omega)
%             run_dpg_notar(trial,folder_save,'basis_degree',basis_degree,'tau_omega',tau_omega)
%             run_tdreg(trial,folder_save,'basis_degree',basis_degree,'tau_omega',tau_omega)
            run_td3(trial,folder_save,'basis_degree',basis_degree,'tau_omega',tau_omega)
%             run_td3reg(trial,folder_save,'basis_degree',basis_degree,'tau_omega',tau_omega)
        end
    end
end
