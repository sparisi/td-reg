close all

for basis_degree = [2, 3]
    for ep_learn = [1, 5]
        for noisy = [false, true]
            folder_save = ['./' num2str(basis_degree) '_' num2str(ep_learn) '_' num2str(noisy) '/'];
            mkdir(folder_save)
            parfor trial = 1 : 50
               try run_reinforce(trial, folder_save, 'ep_learn', ep_learn, 'noisy', noisy, 'basis_degree', basis_degree), catch, folder_save, trial, end
               try run_spg_td(trial, folder_save, 'ep_learn', ep_learn, 'noisy', noisy, 'basis_degree', basis_degree), catch, folder_save, trial, end
               try run_spg_tdreg(trial, folder_save, 'ep_learn', ep_learn, 'noisy', noisy, 'basis_degree', basis_degree), catch, folder_save, trial, end
               try run_spg_mc(trial, folder_save, 'ep_learn', ep_learn, 'noisy', noisy, 'basis_degree', basis_degree), catch, folder_save, trial, end
               try run_spg_mcreg(trial, folder_save, 'ep_learn', ep_learn, 'noisy', noisy, 'basis_degree', basis_degree), catch, folder_save, trial, end
            end
        end
    end
end
