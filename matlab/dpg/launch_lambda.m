close all

for basis_degree = [3]
    for tau_omega = [1]
        folder_save = ['./lambda_' num2str(basis_degree) '_' num2str(tau_omega) '/'];
        mkdir(folder_save)
        for lambda_decay = [0, 0.1, 0.5, 0.9, 0.99, 0.999, 1, 1.001]
            args = {'basis_degree', basis_degree, 'tau_omega', tau_omega, 'lambda_decay', lambda_decay};
            parfor trial = 1 : 50
                run_dpg_reg(trial, folder_save, args{:})
                % edit run_dpg_reg to add lambda_decay to the name of the filename
            end
        end
    end
end
