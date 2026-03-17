function genkwave2(out_h5, Ntrain, Ntest, seed, batch_size)

seed = 0;
batch_size = 32;

cfg.Nx = 128;
cfg.Nz = 256;
cfg.dx = 2e-4;
cfg.dz = 2e-4;
cfg.c0 = 1540;
cfg.rho0 = 1000;
cfg.t_end = 20e-6;
cfg.Ne = 64;
cfg.pitch_m = cfg.dx;
cfg.Ccond = 6;

kgrid = kWaveGrid(cfg.Nx, cfg.dx, cfg.Nz, cfg.dz);
kgrid.makeTime(cfg.c0, [], cfg.t_end);
cfg.kgrid = kgrid;
cfg.Nt = length(kgrid.t_array);
cfg.fs = 1 / kgrid.dt;

probe_z = 5;
pitch_pts = 1;
cx_ula = round(cfg.Nx/2);
x0 = cx_ula - floor(cfg.Ne/2) * pitch_pts;

cfg.sensor.mask = zeros(cfg.Nx, cfg.Nz);
for e = 1:cfg.Ne
    xe = x0 + (e-1) * pitch_pts;
    cfg.sensor.mask(xe, probe_z) = 1;
end
cfg.source.p_mask = cfg.sensor.mask;

[cfg.X, cfg.Z] = ndgrid(1:cfg.Nx, 1:cfg.Nz);
cfg.base_c = cfg.c0 * ones(cfg.Nx, cfg.Nz, 'single');
cfg.base_rho = cfg.rho0 * ones(cfg.Nx, cfg.Nz, 'single');

cfg.inc_r_min = 6;
cfg.inc_r_max = 18;
cfg.inc_margin = cfg.inc_r_max + 12;
cfg.delta_c = -80;
cfg.delta_rho = 20;
cfg.steer_min_deg = -30;
cfg.steer_max_deg = 30;
cfg.seed = seed;
cfg.batch_size = batch_size;

if exist(out_h5, 'file'); delete(out_h5); end
if Ntrain > 0; create_h5_group(out_h5, 'train', Ntrain, cfg); end
if Ntest > 0;  create_h5_group(out_h5, 'test',  Ntest,  cfg); end

if Ntrain > 0; run_batch_split(out_h5, 'train', Ntrain, 0, cfg); end
if Ntest > 0;  run_batch_split(out_h5, 'test',  Ntest, Ntrain, cfg); end

fprintf('done\n');

end

function create_h5_group(h5file, group, N, cfg)
    chunk_N = min(cfg.batch_size, N);
    h5create(h5file, sprintf('/%s/pulse_rf', group), [N, cfg.Nt], 'Datatype','single', 'ChunkSize',[chunk_N, cfg.Nt]);
    h5create(h5file, sprintf('/%s/cond', group), [N, cfg.Ccond], 'Datatype','single', 'ChunkSize',[chunk_N, cfg.Ccond]);
    h5create(h5file, sprintf('/%s/rx_rf', group), [N, cfg.Nt, cfg.Ne], 'Datatype','single', 'ChunkSize',[chunk_N, cfg.Nt, cfg.Ne]);
    h5create(h5file, sprintf('/%s/c_map', group), [N, cfg.Nx, cfg.Nz], 'Datatype','single', 'ChunkSize',[chunk_N, cfg.Nx, cfg.Nz]);
    h5create(h5file, sprintf('/%s/rho_map', group), [N, cfg.Nx, cfg.Nz], 'Datatype','single', 'ChunkSize',[chunk_N, cfg.Nx, cfg.Nz]);
    h5create(h5file, sprintf('/%s/inc_mask', group), [N, cfg.Nx, cfg.Nz], 'Datatype','uint8', 'ChunkSize',[chunk_N, cfg.Nx, cfg.Nz]);
end

function run_batch_split(h5file, group, Ntotal, seed_offset, cfg)
    input_args = {'PMLSize', 16, 'PMLInside', false, 'PlotSim', false};
    x_coords = ((0:cfg.Ne-1) - (cfg.Ne-1)/2) * cfg.pitch_m;
    dt = cfg.kgrid.dt;

    % input_args = {'DataCast','gpuArray-single'};     % ghost: old GPU path
    % sensor_data = gather(sensor_data);                % ghost: unnecessary in serial

    for i = 1:Ntotal
        rng(cfg.seed + seed_offset + i);

        c_map = cfg.base_c;
        rho_map = cfg.base_rho;

        inc_r = randi([cfg.inc_r_min, cfg.inc_r_max]);
        inc_x = randi([cfg.inc_margin, cfg.Nx - cfg.inc_margin]);
        inc_z = randi([cfg.inc_margin + 15, cfg.Nz - cfg.inc_margin]);
        inc_mask = (cfg.X - inc_x).^2 + (cfg.Z - inc_z).^2 <= inc_r^2;

        c_map(inc_mask) = c_map(inc_mask) + cfg.delta_c;
        rho_map(inc_mask) = rho_map(inc_mask) + cfg.delta_rho;

        medium.sound_speed = c_map;
        medium.density = rho_map;

        f0 = 1e6 + 2.65e6 * rand();
        frac_bw = 0.4 + 0.4 * rand();
        nCycles = 2 + randi(4);
        amp = 0.8 + 0.6 * rand();
        phase = 2 * pi * rand();

        base = toneBurst(cfg.fs, f0, nCycles, 'SignalOffset', 0);
        base = base(:) .* hann(length(base));
        base = amp * (base .* cos(phase));

        pulse_rf = zeros(cfg.Nt, 1, 'single');
        L = min(cfg.Nt, length(base));
        pulse_rf(1:L) = single(base(1:L));

        steer = (cfg.steer_min_deg + (cfg.steer_max_deg - cfg.steer_min_deg) * rand()) * pi / 180;
        delays_s = (x_coords * sin(steer)) / cfg.c0;

        source_local = cfg.source;
        source_local.p = zeros(cfg.Ne, cfg.Nt, 'single');
        for e = 1:cfg.Ne
            shift = round(delays_s(e) / dt);
            if shift > 0
                source_local.p(e, 1+shift:end) = pulse_rf(1:cfg.Nt-shift).';
            elseif shift < 0
                sh = -shift;
                source_local.p(e, 1:cfg.Nt-sh) = pulse_rf(1+sh:end).';
            else
                source_local.p(e,:) = pulse_rf.';
            end
        end

        sensor_data = kspaceFirstOrder2D(cfg.kgrid, medium, source_local, cfg.sensor, input_args{:});
        rx_rf = single(sensor_data);
        if size(rx_rf,1) == cfg.Ne && size(rx_rf,2) == cfg.Nt
            rx_rf = rx_rf.';
        end

        pulse_pack = reshape(single(pulse_rf), [1, cfg.Nt]);
        rx_pack = reshape(single(rx_rf), [1, cfg.Nt, cfg.Ne]);

        inc_x_norm = single(inc_x / cfg.Nx);
        inc_z_norm = single(inc_z / cfg.Nz);
        inc_r_norm = single(inc_r / min(cfg.Nx, cfg.Nz));
        cond = single([f0, frac_bw, steer, inc_x_norm, inc_z_norm, inc_r_norm]);

        c_pack = reshape(single(c_map), [1, cfg.Nx, cfg.Nz]);
        rho_pack = reshape(single(rho_map), [1, cfg.Nx, cfg.Nz]);
        mask_pack = reshape(uint8(inc_mask), [1, cfg.Nx, cfg.Nz]);

        h5write(h5file, sprintf('/%s/pulse_rf', group), pulse_pack, [i, 1], [1, cfg.Nt]);
        h5write(h5file, sprintf('/%s/cond', group), reshape(cond, [1, cfg.Ccond]), [i, 1], [1, cfg.Ccond]);
        h5write(h5file, sprintf('/%s/rx_rf', group), rx_pack, [i, 1, 1], [1, cfg.Nt, cfg.Ne]);
        h5write(h5file, sprintf('/%s/c_map', group), c_pack, [i, 1, 1], [1, cfg.Nx, cfg.Nz]);
        h5write(h5file, sprintf('/%s/rho_map', group), rho_pack, [i, 1, 1], [1, cfg.Nx, cfg.Nz]);
        h5write(h5file, sprintf('/%s/inc_mask', group), mask_pack, [i, 1, 1], [1, cfg.Nx, cfg.Nz]);

        if mod(i, 10) == 0 || i == Ntotal
            fprintf('/%s %d/%d\n', group, i, Ntotal);
        end
    end
end
