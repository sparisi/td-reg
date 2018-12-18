close all

for basis_degree = 3
    for ep_learn = [1]
        for noisy = [false]
            folder_save = ['./lambda_' num2str(basis_degree) '_' num2str(ep_learn) '_' num2str(noisy) '/'];
            mkdir(folder_save)
            for lambda_decay = [0, 0.1, 0.5, 0.9, 0.99, 0.999, 1, 1.001]
                parfor trial = 1 : 50
                    try run_spg_tdreg(trial, folder_save, 'ep_learn', ep_learn, 'noisy', noisy, 'basis_degree', basis_degree, 'lambda_decay', lambda_decay), catch, end
                    % edit run_spg_tdreg to add lambda_decay to the name of the filename
                end
            end
        end
    end
end
