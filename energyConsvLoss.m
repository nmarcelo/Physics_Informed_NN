classdef energyConsvLoss < nnet.layer.RegressionLayer
    properties

    end
    methods
        function layer = energyConsvLoss(name)
            layer.Name = name;
            layer.Description = 'MAE + Energy Conservation Loss';
        end

        function loss = forwardLoss(layer, Y, T)
            MAEloss = mean(abs(Y(:)-T(:)));
            N = size(T,2);
            energy_system_pred=-9.81.*cos(Y(1,:)) + (1/2).*(Y(2,:).^2);
            pred_energy = sum(energy_system_pred)/N;
            energy_system_true = -9.8.*cos(T(1,:)) + (1/2).*(T(2,:).^2);
            true_energy = sum(energy_system_true)/N;
            e_factor = 1.0e-5;
            L = 0.5;
            energy_loss = e_factor*(abs(true_energy-pred_energy));
            loss_mag = sqrt(energy_loss^2 + MAEloss^2);
            loss = L*MAEloss*MAEloss/loss_mag + (1-L)*energy_loss*energy_loss/loss_mag;
        end
    end
end