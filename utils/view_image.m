clear; clc;

h5file = "test_1.h5";
group = "/train6000_test600";
sample_idx = 5;

Nx = 128;
Nz = 256;
dx = 2e-4;
dz = 2e-4;
c0 = 1540;
Ne = 64;
probe_z = 5;
t_end = 80e-6;

info_rx = h5info(h5file, group + "/rx_iq");
Nt = info_rx.Dataspace.Size(2);

rx_pack = h5read(h5file, group + "/rx_iq", [sample_idx, 1, 1, 1], [1, Nt, Ne, 2]);
rx_pack = squeeze(rx_pack);
rx = complex(rx_pack(:,:,1), rx_pack(:,:,2));

cond = squeeze(h5read(h5file, group + "/cond", [sample_idx, 1], [1, 6]));
inc_x_norm = cond(4);
inc_z_norm = cond(5);

dt = t_end / (Nt - 1);

pitch_pts = 1;
cx_ula = round(Nx/2);
x0 = cx_ula - floor(Ne/2) * pitch_pts;

x_e = ((x0:(x0+Ne-1)) - 1) * dx;
z_e = (probe_z - 1) * dz * ones(Ne, 1);

x_m = (0:Nx-1) * dx;
z_m = (0:Nz-1) * dz;
bf = zeros(Nz, Nx, 'like', rx);
apo = hann(Ne); apo = apo / sum(apo);

% rx = fft(rx, [], 1);
% z_m = z_m * 1.1; 

for iz = 1:Nz
    z = z_m(iz);
    for ix = 1:Nx
        x = x_m(ix);
        acc = 0;
        for e = 1:Ne
            tau = hypot(x - x_e(e), z - z_e(e)) / c0;
            u = tau / dt + 1;
            i0 = floor(u);
            i0 = max(1, min(i0, Nt-1));
            a = u - i0;
            s0 = rx(i0, e);
            s1 = rx(i0 + 1, e);
            acc = acc + apo(e) * ((1-a) * s0 + a * s1);
        end
        bf(iz, ix) = acc;
    end
end

env = abs(bf);
env = env / (max(env(:)) + 1e-12);
dyn = 55;
bmode = max(20 * log10(env + 1e-6), -dyn);

x_img = ((0:Nx-1) - (Nx-1)/2) * dx;
z_img = (0:Nz-1) * dz;

figure;
imagesc(x_img * 1e3, z_img * 1e3, bmode);
axis image;
colormap gray;
colorbar;
caxis([-dyn, 0]);
xlabel("Lateral x (mm)");
ylabel("Axial z (mm)");
title(sprintf("DAS sample %d", sample_idx));

hold on;
plot((inc_x_norm * Nx * dx - (Nx-1)*dx/2) * 1e3, (inc_z_norm * Nz * dz) * 1e3, "r+", "MarkerSize", 12, "LineWidth", 2);
hold off;