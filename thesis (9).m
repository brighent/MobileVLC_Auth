%% STEP 1. Room Dimensions and Positions

% Define room size
roomLength = 5; % meters
roomWidth = 5; % meters
roomHeight = 4; % meters

% Define dimension of the grid (16x16)
Grid_dimension = [16, 16]; 

% Define initial positions for Alice, Bob, and Eve
alicePos = [0, 0, 2.3];  % Position of Alice (initial)
bobPos = [-0.5, -1.5, 0.5];  % Position of Bob (initial)
evePos = [1, 0, 2];  % Position of Eve (initial)

% Display initial positions in the command window
disp('Initial Positions of Users:');
fprintf('Alice coordinates: [ %.1f, %.1f, %.1f ]\n', alicePos(1), alicePos(2), alicePos(3));
fprintf('Bob coordinates:   [ %.1f, %.1f, %.1f ]\n', bobPos(1), bobPos(2), bobPos(3));
fprintf('Eve coordinates:   [ %.1f, %.1f, %.1f ]\n', evePos(1), evePos(2), evePos(3));
fprintf('\n');

% Create a plot to show positions
figure(1);clf;
hold on;

% Create grid for visualization (16x16 grid)
[xGrid, yGrid] = meshgrid(linspace(-roomLength, roomLength, Grid_dimension(1)), ...
                          linspace(-roomWidth, roomWidth, Grid_dimension(2)));
zGrid = zeros(size(xGrid));  % Create a flat grid at z = 0

% Plot the 16x16 grid (in the XY plane)
surf(xGrid, yGrid, zGrid, 'FaceAlpha', 0.1, 'EdgeColor', 'k', 'DisplayName', 'Grid');

% Plot Alice's position
scatter3(alicePos(1), alicePos(2), alicePos(3), 100, 'd', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'DisplayName', 'Alice');
text(alicePos(1), alicePos(2), alicePos(3), ' Alice', 'FontSize', 12, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontWeight', 'bold');

% Plot Bob's position
scatter3(bobPos(1), bobPos(2), bobPos(3), 100, 's', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g', 'DisplayName', 'Bob');
text(bobPos(1), bobPos(2), bobPos(3), ' Bob', 'FontSize', 12, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontWeight', 'bold');

% Plot Eve's position
scatter3(evePos(1), evePos(2), evePos(3), 100, '^', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b', 'DisplayName', 'Eve');
text(evePos(1), evePos(2), evePos(3), ' Eve', 'FontSize', 12, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontWeight', 'bold');

% Adjust plot limits to include all user positions
xlim([-roomLength, roomLength]);
ylim([-roomWidth, roomWidth]);
zlim([0, roomHeight]);

% Add grid and labels
grid on;
xlabel('X (meters)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Y (meters)', 'FontSize', 12, 'FontWeight', 'bold');
zlabel('Z (meters)', 'FontSize', 12, 'FontWeight', 'bold');
title('User Locations in a 3D Room with 16x16 Grid', 'FontSize', 14, 'FontWeight', 'bold');

% Display the legend
legend('show');

% Set a 3D view
view(3);

% Finalize the plot
hold off;

%% STEP 2. Discrete-Time Signal Generation (S[n])

Nl = 5;                         % Number of signal components
n = 0:100;                      % Discrete time vector (samples)
betas = rand(1, Nl+1);                 % Random weights for each component
Deltas = randi([0, 10], 1, Nl+1);      % Random time shifts (Δi)
fs = 1000;                          % Sampling frequency (Hz)
f = 50;                             % Frequency of sine wave (Hz)

% Signal function g[n - Δi] (shifted sine wave components)
g = @(n, Delta) sin(2 * pi * f * (n - Delta) / fs);

% Initialize the discrete-time signal s[n] with zeros
s_n = zeros(1, length(n));

% Calculate s[n] with the formula
for i = 0:Nl
    s_n = s_n + betas(i+1) * g(n, Deltas(i+1));       % Add each component
end

% Calculate the average of s[n]
average_s_n = mean(s_n);
fprintf('Average of S[n]: %.4f\n', average_s_n);  % Display the average

% Normalize the generated signal to ensure it adheres to Bloch's law
s_n = s_n / max(abs(s_n));  % Normalization to mitigate flickering

fprintf('\n');

% Plot the signal s[n]
figure(2);clf;
hold on;

plot(n, s_n, 'LineWidth', 2);
xlabel('n (Discrete Time)');
ylabel('Amplitude');
title('Discrete-time signal s[n]');
grid on;
hold off;

%% STEP 3. Calculate the Phase Shift

% Constants for Phase Shift Calculation
nr = 1.5;        % Refractive index of reflective surface (e.g., glass)
ni = 1.0;        % Refractive index of air
theta_i = 30;   % Incident angle (degrees)

% Define RGB wavelengths in meters
lambda_R = 650e-9;  % Wavelength for Red light (650 nm)
lambda_G = 532e-9;  % Wavelength for Green light (532 nm)
lambda_B = 475e-9;  % Wavelength for Blue light (475 nm)

% Calculate phase shifts
theta_R = (2 * pi * (nr - ni) * sin(theta_i)) / lambda_R;
theta_G = (2 * pi * (nr - ni) * sin(theta_i)) / lambda_G;
theta_B = (2 * pi * (nr - ni) * sin(theta_i)) / lambda_B;

% Display results
fprintf('Phase shift for Red light: %.2f radians\n', theta_R);
fprintf('Phase shift for Green light: %.2f radians\n', theta_G);
fprintf('Phase shift for Blue light: %.2f radians\n', theta_B);

% Compute Phase Shift (radians) for a single reflection
theta_r_t = (theta_R + theta_G + theta_B) / 3;

% Display result
fprintf('--------------\n'); 
fprintf('phase shift for one RGB LED: %f radians\n', theta_r_t);
fprintf('\n');
fprintf('\n');

%% STEP 4. Reflected Electric Field (Er) with the Calculated Phase Shift

% Incident Electric Field
Ei = 1 + 1j;               % incident electric field in complex form
Er_total = 0;              % Initialize total reflected field
r0 = 0.8;                  % Reflection coefficient
delta_r = pi/4;
N_REs= 20;
Er_R =0;
Er_G = 0;
Er_B = 0;
numConfigs = 10;                  % Number of RIS configurations
loss_factor = 0.9;                % Loss due to environmental factors
Er_total = loss_factor * Er_total;       % Adjust reflected field

% Calculate the reflected electric field with formula
for i = 1:N_REs
    % Calculate reflected electric field for each RGB component
    Er_R = r0 * exp(1j * (theta_R + delta_r)) * Ei;   % for RED
    Er_G = r0 * exp(1j * (theta_G + delta_r)) * Ei;   % for GREEN
    Er_B = r0 * exp(1j * (theta_B + delta_r)) * Ei;   % for BLUE
    
    % Accumulate total reflected field for all RGB components(LED)
    Er_total = Er_total + (Er_R + Er_G + Er_B);
end

% Display results
fprintf('total (Er) for Red light: %.2f + %.2fj\n', real(Er_R), imag(Er_R));
fprintf('total (Er) for Green light: %.2f + %.2fj\n', real(Er_G), imag(Er_G));
fprintf('total (Er) for Blue light: %.2f + %.2fj\n', real(Er_B), imag(Er_B));

% Display the total reflected electric field
fprintf('--------------\n'); 
fprintf('Reflected Electric Field (Er) for one LED : %.2f + %.2fj\n', real(Er_total), imag(Er_total));
fprintf('\n');
fprintf('\n');

Er_total_configs = zeros(1, numConfigs);

% variability in the RIS configurations
% Random delta_r for each configuration to simulate changes
for configIdx = 1:numConfigs         
    delta_r = rand * pi;
    Er_R = r0 * exp(1j * (theta_R + delta_r)) * Ei;
    Er_G = r0 * exp(1j * (theta_G + delta_r)) * Ei;
    Er_B = r0 * exp(1j * (theta_B + delta_r)) * Ei;
    
    Er_total_configs(configIdx) = Er_R + Er_G + Er_B;
end

%% STEP 5. Calculate and Display Distances in the frist position

% Calculate distances between each pair
dist_Alice_Bob = norm(alicePos - bobPos);
dist_Alice_Eve = norm(alicePos - evePos);
dist_Bob_Eve = norm(bobPos - evePos);

% Display distances
fprintf('initial distance between each person: \n');
fprintf('--------------\n'); 
fprintf('Distance between Alice and Bob: %.2f meters\n', dist_Alice_Bob);
fprintf('Distance between Alice and Eve: %.2f meters\n', dist_Alice_Eve);
fprintf('Distance between Bob and Eve: %.2f meters\n', dist_Bob_Eve);
fprintf('\n');

%% STEP 6. Calculate the new distance in the path for one user by using the formula d = d0 + vt (function)

function new_distance = calculate_linear_distance(initial_distance)
    
    % Generate random speed (0 to 5 m/s)
    speed = rand * 5;    
    % Generate random time (0 to 10 seconds)
    time = rand * 10;    
    
    % Calculate the new distance using d = d0 + vt
    new_distance = initial_distance + speed * time;
end

%% STEP 7. calculate the Euclidean distance between two users (function)

function distance = calculate_distance(user1_pos, user2_pos)
    
% Calculate the Euclidean distance between user1 and user2
    distance = norm(user2_pos - user1_pos);
end

%% STEP 8. Calculate LOS (function)

function Hd_0 = calculate_LOS(distance)

    % Constants
    Ar = 1e-4;                            % Receiver area (m^2)
    R = 1;                                % Photodiode responsivity (A/W)
    n = 1.5;                              % Refractive index

    % Convert angles to radians 
    phi = deg2rad(70);
    phi_half = deg2rad(60);
    psi = deg2rad(30);
    psi_FOV = deg2rad(120);

    % Precalculation
    m = -log(2) / log(cos(phi_half));      % Lambertian order

    % Calculate D(psi), the gain of the optical concentrator
    if abs(psi) <= psi_FOV
        D_psi = (n^2) / (sin(psi_FOV)^2);
    else
        D_psi = 0;
    end

    % Calculate LOS channel gain for Bob
    if abs(psi) <= psi_FOV
        Hd_0 = ((Ar * (m + 1) * R) / (2 * pi * distance^2)) * D_psi * cos(phi)^m * cos(psi);
    else
        Hd_0 = 0;  % If the angle of incidence exceeds the FOV, set gain to 0
    end
end


% use the function
fprintf('calculate LOS with initial distance values between each person : \n');
fprintf('--------------\n'); 
fprintf('LOS for Alice and Bob when they are in the frist position: %f\n', calculate_LOS(dist_Alice_Bob));
fprintf('LOS for Alice and Eve when they are in the frist position: %f\n', calculate_LOS(dist_Alice_Eve));
fprintf('LOS for Bob and Eve when they are in the frist position: %f\n', calculate_LOS(dist_Bob_Eve));
fprintf('\n');

%% STEP 9.Calculate NLOS (function)

function dHref_0 = calculate_NLOS(d1B,d2B)
    
% Constants
    Ar = 1e-4;                            % Receiver area (m^2)
    R = 1;                                % Photodiode responsivity (A/W)
    n = 1.5;                              % Refractive index
    dAw = 1e-4;                        
    rho = 0.8; 
    Pt = 1;                            % Transmit power (W)

 % Convert angles to radians 
    phi = deg2rad(70);
    phi_half = deg2rad(60);
    psi = deg2rad(30);
    psi_FOV = deg2rad(120);
    alpha = deg2rad(25);              
    beta = deg2rad(15);               
                        
 % Precalculation
    m = -log(2) / log(cos(phi_half));      % Lambertian order
 
 % Calculate D(psi), the gain of the optical concentrator
    if abs(psi) <= psi_FOV
        D_psi = (n^2) / (sin(psi_FOV)^2);
    else
        D_psi = 0;
    end

 % Calculate dHref(0) based on the NLOS channel gain formula
    if abs(psi) <= psi_FOV
        dHref_0 = ((Ar * (m + 1) * R) / (2 * pi^2 * d1B^2 * d2B^2)) * ...
                 D_psi * Pt * rho * dAw * cos(phi)^m * cos(alpha) * cos(beta) * cos(psi);
    else
        dHref_0 = 0;
    end

    % Multiply output for better readability
    dHref_0 = dHref_0 * 1e10; % Scale factor

end
   

% use the function
fprintf('calculate NLOS with initial distance values between each person : \n');
fprintf('--------------\n'); 
fprintf('NLOS for Alice-Bob and Alice-Eve when they are in the first position: %.10f\n', calculate_NLOS(dist_Alice_Bob , dist_Alice_Eve));
fprintf('NLOS for Alice-Eve and Bob-Eve when they are in the first position: %.10f\n', calculate_NLOS(dist_Alice_Eve , dist_Bob_Eve));
fprintf('NLOS for Alice-Bob and Bob-Eve when they are in the first position: %.10f\n', calculate_NLOS(dist_Alice_Bob , dist_Bob_Eve));
fprintf('\n');



% use the NLOS vs DRIS
% Parameters for NLOS calculation
Pt = 1;                            % Transmit power (W)
dAw = 1e-4;                        % Micro surface area
alpha = deg2rad(25);               % Incidence angle at RIS (in radians)
beta = deg2rad(15);                % Radiation angle at PD (in radians)
phi = deg2rad(30);                 % Angle of irradiance (in radians)
psi = deg2rad(20);                 % Angle of incidence (in radians)
phi_half = deg2rad(70);            % Half-power angle of LED (in radians)
psi_FOV = deg2rad(120);             % Field of View (FOV) of the receiver (in radians)
n = 1.5;                           % Refractive index
rho = 0.8;                         % Reflection coefficient of RIS
m = -log(2) / log(cos(phi_half));  % Lambertian order for half-angle
D = n^2 / sin(psi_FOV)^2;           % Optical concentrator gain
d1B = 2;                           % Distance between LED and RIS (m)
d2B = 1.5;                         % Distance between RIS and PD (m)
Ar = 1e-4;                         % Receiver area (m^2)
R = 1;                             % Photodiode responsivity (A/W)

% Calculate NLOS Gain
N_REs = 20;
H_NLOS_total = 0;  % Initialize total NLOS gain

for i = 1:N_REs
    % Generate random position for each DRIS element within room bounds
    drisPos = [rand() * roomLength, rand() * roomWidth, rand() * roomHeight];
    
    % Store the position in the array
    drisPositions(i, :) = drisPos;
    
    % Calculate distances: Alice-to-DRIS (d1B) and DRIS-to-Bob (d2B)
    d1B = norm(alicePos - drisPos);   % Distance from Alice to DRIS element
    d2B = norm(bobPos - drisPos);     % Distance from DRIS element to Bob
    
    % Calculate NLOS gain for this DRIS element
    if abs(psi) <= psi_FOV
        H_NLOS = ((Ar * (m + 1) * R) / (2 * pi^2 * d1B^2 * d2B^2)) * D * Pt * rho * dAw * cosd(phi)^m * cosd(alpha) * cosd(beta) * cosd(psi);
    else
        H_NLOS = 0;  % Gain is zero if incidence angle exceeds FOV
    end
    
    % Accumulate NLOS gain from each DRIS element
    H_NLOS_total = H_NLOS_total + H_NLOS;
end

% Plot DRIS positions with transparency
figure(3);clf;
scatter3(drisPositions(:, 1), drisPositions(:, 2), drisPositions(:, 3), 50, 'm', 'filled', 'DisplayName', 'DRIS Elements', 'MarkerFaceAlpha', 0.6);
hold on;

% Plot Alice's position
scatter3(alicePos(1), alicePos(2), alicePos(3), 100, 'd', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'DisplayName', 'Alice');
text(alicePos(1), alicePos(2), alicePos(3), ' Alice', 'FontSize', 12, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontWeight', 'bold');

% Plot Bob's position
scatter3(bobPos(1), bobPos(2), bobPos(3), 100, 's', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g', 'DisplayName', 'Bob');
text(bobPos(1), bobPos(2), bobPos(3), ' Bob', 'FontSize', 12, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontWeight', 'bold');

% Plot Eve's position
scatter3(evePos(1), evePos(2), evePos(3), 100, '^', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b', 'DisplayName', 'Eve');
text(evePos(1), evePos(2), evePos(3), ' Eve', 'FontSize', 12, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontWeight', 'bold');

% Finalize the plot
xlabel('X (meters)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Y (meters)', 'FontSize', 12, 'FontWeight', 'bold');
zlabel('Z (meters)', 'FontSize', 12, 'FontWeight', 'bold');
title('3D DRIS Positions and User Locations', 'FontSize', 14, 'FontWeight', 'bold');
legend show;
grid on;

% Set viewing angle for better 3D visualization
view(30, 20); % Adjust as needed for your dataset
hold off;

% Multiply output for better readability
H_NLOS_total = H_NLOS_total * 1e10; % Scale factor

% Display total NLOS gain
fprintf('Total NLOS Gain: %.4f\n', H_NLOS_total);
fprintf('\n');

%% STEP 10. Define blockage factor B(t) (function)

function [Pr_full, Pr_partial, Pr_none] = calculate_received_power(Hd_t)

    Pt = 1;

    % Define blockage factors
    B_full = 0;       % Full blockage
    B_partial = 0.5;  % Partial blockage
    B_none = 1;       % No blockage

    % Calculate received power for each blockage condition
    Pr_full = Pt * Hd_t * B_full;       % Power with full blockage
    Pr_partial = Pt * Hd_t * B_partial; % Power with partial blockage
    Pr_none = Pt * Hd_t * B_none;       % Power with no blockage

    % Display the result for B(t)
    fprintf('Amount of full blockage (B(t) = 0) is : %f\n', Pr_full);
    fprintf('Amount of partial blockage (B(t) = 0.5) is : %f\n', Pr_partial);
    fprintf('Amount of no blockage (B(t) = 1) is : %f\n', Pr_none);
    fprintf('\n');
end

%% STEP 11. Lambertian Channel Model Implementation for Moving Characters

% Room dimensions
roomLength = 5; % meters
roomWidth = 5; % meters
roomHeight = 4; % meters

alicePos = [0, 0, 2.3];  % Position of Alice (initial)
bobPos = [-0.5, -1.5, 0.5];  % Position of Bob (initial)
evePos = [1, 0, 2];  % Position of Eve (initial)

% Fixed total number of steps to 20
numSteps = 20;
timeStep = 0.5;  % Fixed time step of 0.5 seconds

% Initialize arrays to store positions over time
aliceTrajectory = zeros(numSteps, 3);
bobTrajectory = zeros(numSteps, 3);
eveTrajectory = zeros(numSteps, 3);

% Set initial positions
aliceTrajectory(1, :) = alicePos;
bobTrajectory(1, :) = bobPos;
eveTrajectory(1, :) = evePos;

% Calculate initial distances
d_alice_bob = calculate_distance(alicePos, bobPos);
d_bob_eve = calculate_distance(bobPos, evePos);
d_alice_eve = calculate_distance(alicePos, evePos);

% Initialize arrays to store LOS and NLOS values
LOS_alice_bob = zeros(numSteps, 1);
LOS_alice_eve = zeros(numSteps, 1);
LOS_bob_eve = zeros(numSteps, 1);

NLOS_alice_bob = zeros(numSteps, 1);
NLOS_alice_eve = zeros(numSteps, 1);
NLOS_bob_eve = zeros(numSteps, 1);

savedAlicePositions = zeros(numSteps, 3);
savedBobPositions = zeros(numSteps, 3);
savedEvePositions = zeros(numSteps, 3);

% Loop through each step and update user positions and distances
for t = 1:numSteps
    if t == 1 
        fprintf('---------------------------------------\n');
        fprintf('What if our user move ? \n');
        fprintf('Step %d\n', t);
        fprintf('  The first distances between users:\n');
        fprintf('  Distance between Alice-Bob: %.2f meters\n', d_alice_bob);
        fprintf('  Distance between Bob-Eve: %.2f meters\n', d_bob_eve);
        fprintf('  Distance between Alice-Eve: %.2f meters\n', d_alice_eve);
        fprintf('\n');

    else
        fprintf('Step %d\n', t);
        
        % Calculate new distance for each user with calculate_linear_distance
        new_distance_alice = calculate_linear_distance(aliceTrajectory(t-1, :));
        new_distance_bob = calculate_linear_distance(bobTrajectory(t-1, :));
        new_distance_eve = calculate_linear_distance(eveTrajectory(t-1, :));

        % Generate random directions for Alice, Bob, and Eve
        aliceDirection = rand(1, 3) - 0.5;
        bobDirection = rand(1, 3) - 0.5;
        eveDirection = rand(1, 3) - 0.5;
    
        % Normalize direction vectors
        aliceDirection = aliceDirection / norm(aliceDirection);
        bobDirection = bobDirection / norm(bobDirection);
        eveDirection = eveDirection / norm(eveDirection);
    
        % Update positions based on new distance
        aliceTrajectory(t, :) = aliceTrajectory(t-1, :) + new_distance_alice .* aliceDirection .* timeStep;
        bobTrajectory(t, :) = bobTrajectory(t-1, :) + new_distance_bob .* bobDirection .* timeStep;
        eveTrajectory(t, :) = eveTrajectory(t-1, :) + new_distance_eve .* eveDirection .* timeStep;
    
        % Ensure positions stay within room boundaries
        aliceTrajectory(t, :) = max(min(aliceTrajectory(t, :), [roomLength, roomWidth, roomHeight]), [0, 0, 0]);
        bobTrajectory(t, :) = max(min(bobTrajectory(t, :), [roomLength, roomWidth, roomHeight]), [0, 0, 0]);
        eveTrajectory(t, :) = max(min(eveTrajectory(t, :), [roomLength, roomWidth, roomHeight]), [0, 0, 0]);
    
        % Save the new positions
        savedAlicePositions(t, :) = aliceTrajectory(t, :);
        savedBobPositions(t, :) = bobTrajectory(t, :);
        savedEvePositions(t, :) = eveTrajectory(t, :);
    
        % Print new positions of the users
        fprintf('New Position of Alice: [%.2f, %.2f, %.2f]\n', savedAlicePositions(t, 1), savedAlicePositions(t, 2), savedAlicePositions(t, 3));
        fprintf('New Position of Bob: [%.2f, %.2f, %.2f]\n', savedBobPositions(t, 1), savedBobPositions(t, 2), savedBobPositions(t, 3));
        fprintf('New Position of Eve: [%.2f, %.2f, %.2f]\n', savedEvePositions(t, 1), savedEvePositions(t, 2), savedEvePositions(t, 3));
        fprintf('\n');
    
        % Calculate LOS using the predefined function
        Hd_alice_bob = calculate_LOS(d_alice_bob); 
        Hd_alice_eve = calculate_LOS(d_alice_eve); 
        Hd_bob_eve = calculate_LOS(d_bob_eve);
    
        % Saving LOS inside the array
        LOS_alice_bob(t) = calculate_LOS(d_alice_bob);
        LOS_alice_eve(t) = calculate_LOS(d_alice_eve);
        LOS_bob_eve(t) = calculate_LOS(d_bob_eve);
    
        % print
        fprintf('LOS at this position for Alice-Bob: %f\n', Hd_alice_bob);
        fprintf('LOS at this position for Alice-Eve: %f\n', Hd_alice_eve);
        fprintf('LOS at this position for Bob-Eve: %f\n', Hd_bob_eve);
        fprintf('\n');
    
        % Calculate NLOS using the predefined function
        fprintf('NLOS at this position for Alice-Bob: %f\n', calculate_NLOS(d_alice_bob, d_bob_eve));
        fprintf('NLOS at this position for Alice-Eve: %f\n', calculate_NLOS(d_alice_eve, d_bob_eve));
        fprintf('NLOS at this position for Bob-Eve: %f\n', calculate_NLOS(d_bob_eve, d_alice_eve));
        fprintf('\n');
    
        % Saving NlOS inside an array 
        NLOS_alice_bob(t) = calculate_NLOS(d_alice_bob, d_bob_eve);
        NLOS_alice_eve(t) = calculate_NLOS(d_alice_eve, d_bob_eve);
        NLOS_bob_eve(t) = calculate_NLOS(d_bob_eve, d_alice_eve);
    
        % Generate blockage based on the random variable
        fprintf('Blockage for Alice and Bob: \n');
        calculate_received_power(Hd_alice_bob);
        fprintf('Blockage for Alice and Eve: \n');
        calculate_received_power(Hd_alice_eve);
        fprintf('Blockage for Bob and Eve: \n');
        calculate_received_power(Hd_bob_eve);
        fprintf('---------------------------------------\n');
    end 
end 

% Plotting Results
% Plotting the new positions of each user over time
figure(4);clf;
plot3(savedAlicePositions(:, 1), savedAlicePositions(:, 2), savedAlicePositions(:, 3), '-o', 'DisplayName', 'Alice Positions');
hold on;
plot3(savedBobPositions(:, 1), savedBobPositions(:, 2), savedBobPositions(:, 3), '-s', 'DisplayName', 'Bob Positions');
plot3(savedEvePositions(:, 1), savedEvePositions(:, 2), savedEvePositions(:, 3), '-^', 'DisplayName', 'Eve Positions');
xlabel('X Position');
ylabel('Y Position');
zlabel('Z Position');
title('Users Positions over Time');
legend;
grid on;
hold off;

% Plotting the LOS for each user pair
figure(5);clf;
plot(1:numSteps, LOS_alice_bob, '-o', 'DisplayName', 'LOS Alice-Bob');
hold on;
plot(1:numSteps, LOS_alice_eve, '-s', 'DisplayName', 'LOS Alice-Eve');
plot(1:numSteps, LOS_bob_eve, '-^', 'DisplayName', 'LOS Bob-Eve');
xlabel('Time Step');
ylabel('LOS Value');
title('LOS Values over Time');
legend;
grid on;
hold off;

% Plotting the NLOS for each user pair
figure(6);clf;
plot(1:numSteps, NLOS_alice_bob, '-o', 'DisplayName', 'NLOS Alice-Bob');
hold on;
plot(1:numSteps, NLOS_alice_eve, '-s', 'DisplayName', 'NLOS Alice-Eve');
plot(1:numSteps, NLOS_bob_eve, '-^', 'DisplayName', 'NLOS Bob-Eve');
xlabel('Time Step');
ylabel('NLOS Value');
title('NLOS Values over Time');
legend;
grid on;
hold off;

% %% STEP 11. Challenge-Response Based PLA with Actual Challenge from Bob
% % Parameters
% K = 10;               % Number of pilot signals (time slots)
% M = 5;                % Number of configurations for each RE
% threshold = 0.03;     % Threshold for authentication
% Hd_t_phase = 1;       % Channel response phase factor (placeholder value)
% 
% % Initialize dictionary for storing reference signals in enrollment phase
% dictionary = zeros(K, M);         % Stores expected signals for each configuration
% received_signals = zeros(K, 1);   % To store received signals during authentication
% deviations = zeros(K, 1);         % To store deviation values
% 
% % --- Enrollment Phase ---
% fprintf('--- Enrollment Phase ---\n');
% for k = 1:K
%     for m = 1:M
%         % Generate pilot signal with some noise
%         pilot_signal = Hd_t_phase * (rand > 0.5) + randn * 0.01;
%         dictionary(k, m) = pilot_signal;  % Store signal in Bob's dictionary
%         fprintf('Pilot signal for time slot %d, Config %d: %.4f\n', k, m, pilot_signal);
%     end
%     fprintf('\n');
% end
% 
% % --- Authentication Phase with Eve's Attack ---
% fprintf('\n--- Authentication Phase ---\n');
% 
% for attempt = 1:K
%     % Step 1: Bob sends a random binary challenge and chooses a random RE index
%     challenge = rand > 0.5;        % Random binary challenge (0 or 1)
%     RE_index = randi(M);           % Random RE configuration index
% 
%     % Step 2: Alice generates a response based on the challenge and dictionary
%     expected_response = dictionary(attempt, RE_index) * (challenge + 1);  
%     received_signal = expected_response + randn * 0.01;  % Add noise to simulate conditions
% 
%     % Step 3: Eve attempts to impersonate Alice
%     eve_estimation_noise = 0.05;  % Eve's estimation error
%     estimated_channel_eve = dictionary(attempt, randi(M)) + randn * eve_estimation_noise;
%     fake_response = estimated_channel_eve * (challenge + 1) + randn * 0.02;
% 
%     % Store received signal and calculate deviation from expected response
%     received_signals(attempt) = received_signal; 
%     deviation_alice = abs(received_signal - expected_response);
%     deviation_eve = abs(fake_response - expected_response);
%     deviations(attempt) = deviation_alice;
% 
%     % Display the challenge-response process
%     fprintf('Attempt %d:\nBob sends challenge: %d\n', attempt, challenge);
%     fprintf('Expected response: %.4f\nReceived signal (Alice): %.4f\nFake response (Eve): %.4f\n', expected_response, received_signal, fake_response);
% 
%     % Step 4: Bob checks if the deviation is within the threshold
%     if deviation_alice < threshold
%         fprintf('✅ Authentication successful: Alice detected.\n');
%     elseif deviation_eve < threshold
%         fprintf('⚠️ WARNING: Eve impersonation detected! (Eve fooled Bob)\n');
%     else
%         fprintf('❌ Authentication failed: Unrecognized transmission.\n');
%     end
%     fprintf('\n');
% end
% 
% % --- GLRT-Based Detection of Known Channels ---
% 
% % Parameters for Hypothesis Testing
% gamma = 0.03;               % Threshold for GLRT
% K = 10;                     % Number of pilot signals
% Hd_t_phase = 1;             % Channel response phase factor
% deviation = zeros(1, K);    % Storage for deviations
% 
% 
% % Initialize arrays to store received signals and expected channels
% received_signals = zeros(1, K); 
% expected_channels = zeros(1, K); 
% 
% fprintf('--------------------------------------\n');
% fprintf('--- GLRT Signal Detection ---\n');
% 
% % Step 1: Simulate the reception of pilot signals
% for k = 1:K
%     challenge = randi([0, 1]);
%     pilot_signal = randi([0, 1]);
%     expected_channels(k) = Hd_t_phase * pilot_signal * (challenge + 1);
%     noise = randn * 0.01; 
%     received_signals(k) = expected_channels(k) + noise;
%     fprintf('Time Slot %d | Challenge: %d | Expected: %.4f | Received: %.4f\n', ...
%         k, challenge, expected_channels(k), received_signals(k));
% end
% 
% % Step 2: Calculate deviations
% for k = 1:K
%     deviation(k) = abs(received_signals(k) - expected_channels(k));
% end
% 
% % Step 3: Calculate average deviation
% average_deviation = mean(deviation);
% 
% % Step 4: GLRT Hypothesis Testing
% if average_deviation < gamma
%     fprintf('GLRT Result: Authentication Successful - Legitimate Transmission Detected.\n');
% else
%     fprintf('GLRT Result: Authentication Failed - Potential Impersonation Detected.\n');
% end

%% STEP 12. NUMERICAL EVALUATION WITH GLRT-BASED DETECTION

num_trials = 1000;  % Number of Monte Carlo experiments
threshold = 0.03;   % Detection threshold (gamma)
P_FA_count = 0;     % Counter for False Alarms
P_MD_count = 0;     % Counter for Missed Detections
M = 160;              % Number of RIS configurations
N = 20;              % Number of DRIS elements per configuration
K = 4;             % Number of pilot signals (time slots)

% --- Enrollment Phase ---
dictionary = zeros(K, M * N);  % Stores expected signals for each configuration
for k = 1:K
    for m = 1:(M * N)
        dictionary(k, m) = randn * 0.01 + (rand > 0.5); % Generate pilot signals
    end
end

% Monte Carlo Simulation Loop
for i = 1:num_trials
    % Step 1: Bob sends a random challenge and selects a random RE configuration
    challenge = rand > 0.5;        % Random binary challenge (0 or 1)
    RE_index = randi(M);           % Random RIS element configuration
    RE_elements = (RE_index - 1) * N + (1:N); % Get indices for selected RIS elements
    
    % Step 2: Alice responds based on the enrollment dictionary
    expected_response = sum(dictionary(randi(K), RE_elements)) * (challenge + 1);
    received_signal = expected_response + randn * 0.01; % Add noise
    
    % Step 3: Eve attempts to impersonate Alice
    eve_estimation_noise = 0.05;
    estimated_channel_eve = sum(dictionary(randi(K), randi(M * N, 1, N))) + randn * eve_estimation_noise;
    fake_response = estimated_channel_eve * (challenge + 1) + randn * 0.02;
    
    % Compute deviations
    deviation_alice = abs(received_signal - expected_response);
    deviation_eve = abs(fake_response - expected_response);
    
    % Step 4: Apply Heaviside step function correctly
    H_eve = deviation_eve < threshold;   % Eve successfully impersonates Alice (False Alarm)
    H_alice = deviation_alice < threshold; % Alice is correctly authenticated
    
    % Compute probabilities based on correct conditions
    if H_eve == 1  % False Alarm: Eve is mistakenly authenticated
        P_FA_count = P_FA_count + 1;
    end
    if H_alice == 0  % Missed Detection: Alice is mistakenly rejected
        P_MD_count = P_MD_count + 1;
    end
end

% Calculate probabilities
P_FA = P_FA_count / num_trials;
P_MD = P_MD_count / num_trials;

fprintf('\n');
fprintf('----------------------------------------------\n');
fprintf('Probability of False Alarm (PFA): %.4f\n', P_FA);
fprintf('Probability of Missed Detection (PMD): %.4f\n', P_MD);
fprintf('\n');


% Plot Monte Carlo Results for PFA and PMD
figure(7);clf;
bar([P_FA, P_MD], 'FaceColor', [0.2 0.7 0.3]);
set(gca, 'XTickLabel', {'PFA', 'PMD'});
ylabel('Probability');
title('Monte Carlo Simulation Results for PFA and PMD');
grid on;

% --- GLRT-Based Detection of Known Channels ---

gamma = 0.03;               % Threshold for GLRT
deviation = zeros(1, K);    % Storage for deviations
received_signals = zeros(1, K); 
expected_channels = zeros(1, K); 

fprintf('--------------------------------------\n');
fprintf('--- GLRT Signal Detection ---\n');

% Step 1: Simulate the reception of pilot signals
for k = 1:K
    challenge = randi([0, 1]);
    pilot_signal = randi([0, 1]);
    expected_channels(k) = sum(dictionary(k, 1:N)) * (challenge + 1);
    noise = randn * 0.01; 
    received_signals(k) = expected_channels(k) + noise;
    fprintf('Time Slot %d | Challenge: %d | Expected: %.4f | Received: %.4f\n', ...
        k, challenge, expected_channels(k), received_signals(k));
end

% Step 2: Calculate deviations
for k = 1:K
    deviation(k) = abs(received_signals(k) - expected_channels(k));
end

% Step 3: Calculate average deviation
average_deviation = mean(deviation);

% Step 4: GLRT Hypothesis Testing
if average_deviation < gamma
    fprintf('GLRT Result: Authentication Successful - Legitimate Transmission Detected.\n');
else
    fprintf('GLRT Result: Authentication Failed - Potential Impersonation Detected.\n');
end

%% STEP 13. NUMERICAL EVALUATION For diffrent number of K

num_trials = 1000;  % Number of Monte Carlo experiments
threshold = 0.03;   % Detection threshold (gamma)
P_FA_count = 0;     % Counter for False Alarms
P_MD_count = 0;     % Counter for Missed Detections
M = 160;              % Number of RIS configurations
N = 20;              % Number of DRIS elements per configuration
K_values = 1:5;    % Different values of K (pilot signals)

P_FA_results = zeros(size(K_values));
P_MD_results = zeros(size(K_values));

% Fix random seed for reproducibility
rng(42);

for idx = 1:length(K_values)
    K = K_values(idx);
    dictionary = zeros(K, M * N);  % Stores expected signals for each configuration
    
    % --- Enrollment Phase ---
    fixed_indices = randi(K, num_trials, 1); % Precompute indices for consistency
    fixed_RE_indices = randi(M, num_trials, 1);
    
    for k = 1:K
        for m = 1:(M * N)
            dictionary(k, m) = randn * 0.01 + (rand > 0.5); % Generate pilot signals
        end
    end
    
    P_FA_count = 0;
    P_MD_count = 0;
    
    % Monte Carlo Simulation Loop
    for i = 1:num_trials
        % Step 1: Bob sends a random challenge and selects a fixed RE configuration
        challenge = rand > 0.5;        % Random binary challenge (0 or 1)
        RE_index = fixed_RE_indices(i);           % Fixed RIS element configuration
        RE_elements = (RE_index - 1) * N + (1:N); % Get indices for selected RIS elements
        
        % Step 2: Alice responds based on the enrollment dictionary
        expected_response = sum(dictionary(fixed_indices(i), RE_elements)) * (challenge + 1);
        received_signal = expected_response + randn * 0.01; % Add noise
        
        % Step 3: Eve attempts to impersonate Alice
        eve_estimation_noise = 0.05;
        estimated_channel_eve = sum(dictionary(fixed_indices(i), randi(M * N, 1, N))) + randn * eve_estimation_noise;
        fake_response = estimated_channel_eve * (challenge + 1) + randn * 0.02;
        
        % Compute deviations
        deviation_alice = abs(received_signal - expected_response);
        deviation_eve = abs(fake_response - expected_response);
        
        % Step 4: Apply Heaviside step function correctly
        H_eve = deviation_eve < threshold;   % Eve successfully impersonates Alice (False Alarm)
        H_alice = deviation_alice < threshold; % Alice is correctly authenticated
        
        % Compute probabilities based on correct conditions
        if H_eve == 1  % False Alarm: Eve is mistakenly authenticated
            P_FA_count = P_FA_count + 1;
        end
        if H_alice == 0  % Missed Detection: Alice is mistakenly rejected
            P_MD_count = P_MD_count + 1;
        end
    end
    
    % Calculate probabilities for this K
    P_FA_results(idx) = P_FA_count / num_trials;
    P_MD_results(idx) = P_MD_count / num_trials;
end

% Plot Monte Carlo Results for PFA and PMD
figure (8);
hold on;
plot(K_values, P_FA_results, '-o', 'LineWidth', 2, 'Color', 'r');
plot(K_values, P_MD_results, '-s', 'LineWidth', 2, 'Color', 'b');
legend('P_{FA}', 'P_{MD}');
xlabel('Number of Pilot Signals (K)');
ylabel('Probability');
title('Monte Carlo Simulation: PFA and PMD vs. K');
grid on;
hold off;

%% STEP 14. NUMERICAL EVALUATION FOR DIFFERENT NUMBER OF N

num_trials = 1000;  % Number of Monte Carlo experiments
threshold = 0.03;   % Detection threshold (gamma)
P_FA_count = 0;     % Counter for False Alarms
P_MD_count = 0;     % Counter for Missed Detections
M = 160;            % Number of RIS configurations
K = 4;              % Fixed number of pilot signals
N_values = 1:20;    % Different values of N (number of DRIS elements)

P_FA_results = zeros(size(N_values));
P_MD_results = zeros(size(N_values));

% Fix random seed for reproducibility
rng(42);

for idx = 1:length(N_values)
    N = N_values(idx);
    dictionary = zeros(K, M * N);  % Stores expected signals for each configuration
    
    % --- Enrollment Phase ---
    fixed_indices = randi(K, num_trials, 1); % Precompute indices for consistency
    fixed_RE_indices = randi(M, num_trials, 1);
    
    for k = 1:K
        for m = 1:(M * N)
            dictionary(k, m) = randn * 0.01 + (rand > 0.5); % Generate pilot signals
        end
    end
    
    P_FA_count = 0;
    P_MD_count = 0;
    
    % Monte Carlo Simulation Loop
    for i = 1:num_trials
        % Step 1: Bob sends a random challenge and selects a fixed RE configuration
        challenge = rand > 0.5;        % Random binary challenge (0 or 1)
        RE_index = fixed_RE_indices(i);           % Fixed RIS element configuration
        RE_elements = (RE_index - 1) * N + (1:N); % Get indices for selected RIS elements
        
        % Step 2: Alice responds based on the enrollment dictionary
        expected_response = sum(dictionary(fixed_indices(i), RE_elements)) / sqrt(N) * (challenge + 1);
        received_signal = expected_response + randn * 0.01 / sqrt(N); % Noise scaling
        
        % Step 3: Eve attempts to impersonate Alice
        eve_estimation_noise = 0.05;
        estimated_channel_eve = sum(dictionary(fixed_indices(i), randi(M * N, 1, N))) / sqrt(N) + randn * eve_estimation_noise;
        fake_response = estimated_channel_eve * (challenge + 1) + randn * 0.02;
        
        % Compute deviations
        deviation_alice = abs(received_signal - expected_response);
        deviation_eve = abs(fake_response - expected_response);
        
        % Step 4: Apply Heaviside step function correctly
        H_eve = deviation_eve < threshold;   % Eve successfully impersonates Alice (False Alarm)
        H_alice = deviation_alice < threshold; % Alice is correctly authenticated
        
        % Compute probabilities based on correct conditions
        if H_eve == 1  % False Alarm: Eve is mistakenly authenticated
            P_FA_count = P_FA_count + 1;
        end
        if H_alice == 0  % Missed Detection: Alice is mistakenly rejected
            P_MD_count = P_MD_count + 1;
        end
    end
    
    % Calculate probabilities for this N
    P_FA_results(idx) = P_FA_count / num_trials;
    P_MD_results(idx) = P_MD_count / num_trials;
end

% Plot Monte Carlo Results for PFA and PMD
figure (9);
hold on;
plot(N_values, P_FA_results, '-o', 'LineWidth', 2, 'Color', 'r');
plot(N_values, P_MD_results, '-s', 'LineWidth', 2, 'Color', 'b');
legend('P_{FA}', 'P_{MD}');
xlabel('Number of DRIS Elements (N)');
ylabel('Probability');
title('Monte Carlo Simulation: PFA and PMD vs. N');
grid on;
hold off;

%% STEP 15. Impact of SNR on Probability of Successful Attack (P_Eve)

SNR_values = 0:5:20;                 % SNR range from 0 to 30 dB
P_Eve_success = zeros(length(SNR_values), 1);  % Array for Eve's success probabilities
num_trials = 1000;                   % Number of trials for each SNR level
threshold = 0.05;                    % Example threshold based on legitimate signal strength
Hd_t = 1;                            % Hypothetical channel gain of legitimate path (Alice-Bob)
N_REs = 20;                          % Number of REs (reflective elements)
M_rotations = 160;                   % Number of possible rotations per RE

fprintf('\n');
fprintf('--- Impact of SNR on Probability of Successful Attack (P_{Eve}) ---\n');
fprintf('SNR (dB) | Probability of Eve Success (P_{Eve})\n');
fprintf('----------------------------------------------\n');

% Parameters for theoretical P_Eve calculation
C = N_REs * M_rotations;             % Calculate the total configuration space
K = 4;                               % Length of the sequence for authentication
P_Eve_theoretical = 1 / nchoosek(C + K - 1, K);  % Theoretical probability

for idx = 1:length(SNR_values)
    SNR = SNR_values(idx);
    noise_variance = 10^(-SNR / 10);
    Eve_success_count = 0;

    % Simulate trials for Eve's success probability at each SNR
    for i = 1:num_trials
        noise = sqrt(noise_variance) * randn;   % Add Gaussian noise
        received_signal = Hd_t + noise;

        % Dynamic margin for success, adjust based on noise
        dynamic_margin = 0.1 * sqrt(noise_variance); % Adjust margin with SNR

        % Eve succeeds if received signal is close enough to the threshold
        if abs(received_signal - threshold) < dynamic_margin
            Eve_success_count = Eve_success_count + 1;
        end
    end
    
    % Calculate success rate for current SNR
    P_Eve_success(idx) = Eve_success_count / num_trials;

    % Print each result in the command window
    fprintf('   %2d dB   |           %.4f\n', SNR, P_Eve_success(idx));
end

fprintf('----------------------------------------------\n');

% Plot Probability of Eve's Success vs. SNR with Theoretical Bound
figure(10);clf;
hold on;
plot(SNR_values, P_Eve_success, '-o', 'LineWidth', 2, 'DisplayName', 'Simulated P_{Eve}');
yline(P_Eve_theoretical, '--r', 'LineWidth', 1.5, 'DisplayName', 'Theoretical P_{Eve}');
xlabel('SNR (dB)');
ylabel('Probability of Eve Success');
title('Impact of SNR on Probability of Successful Attack');
legend;
grid on;
hold off;


% Probability of a Successful Attack without Noise and Perfect Channel Estimation

K_values = 1:20;  % Range of K values
C_values = 2:6;   % Range of C values

figure(11);clf;
hold on;

% Define colors and markers to match the image
colors = {'#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#0072BD'}; 
markers = {'x', 'v', '^', 's', '*'};

for i = 1:length(C_values)
    C = C_values(i);
    P_Eve_theoretical = arrayfun(@(K) 1 / nchoosek(C + K - 1, K), K_values);
    
    % Plot with different styles
    semilogy(K_values, P_Eve_theoretical, 'Color', colors{i}, ...
        'Marker', markers{i}, 'LineStyle', '-', 'LineWidth', 2, 'MarkerSize', 8);
end

grid on;
set(gca, 'YScale', 'log'); % Log scale for probability
set(gca, 'FontSize', 14); % Set axis font size for readability
axis([0 20 1e-5 1]); % Ensure proper axis limits to match the image

% Axis labels
xlabel('$K$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$P_{\mathrm{Eve, succ}}$', 'Interpreter', 'latex', 'FontSize', 16);
title('Probability of a Successful Attack vs. $K$', 'Interpreter', 'latex', 'FontSize', 16);

% Add legend with LaTeX formatting
legend_labels = arrayfun(@(C) sprintf('$C = %d$', C), C_values, 'UniformOutput', false);
legend(legend_labels, 'Interpreter', 'latex', 'Location', 'SouthWest');

hold off;


%% STEP 16. Enhanced Comparison Between RIS vs DRIS with Lambertian Model & Authentication

Pt = 1;  % Transmit power (W)
N_REs = 20;  % Number of Reflecting Elements (REs)
K = 4;  % Number of time slots for authentication
M = 160;  % Number of configurations per RE
sigma_noise = 1e-7;  % Noise power spectral density

gamma = 0.05;  % Threshold for False Alarm and Misdetection
numIterations = 1000;  % Monte Carlo iterations for DRIS
numSimulations = 1000;  % Monte Carlo runs for FA/MD estimation

% Room setup
roomLength = 5; roomWidth = 5; roomHeight = 4;
alicePos = [0, 0, 2.3];
bobPos = [-0.5, -1.5, 0.5];

% Average distance between Alice and Bob
dist_Alice_Bob = norm(alicePos - bobPos);
H_RIS = Pt / (dist_Alice_Bob^2);  % RIS Channel Gain

% DRIS Channel Gain Calculation using Lambertian Model
H_DRIS_total = 0;
for iter = 1:numIterations
    H_DRIS_iter = 0;
    for i = 1:N_REs
        drisPos = [rand() * roomLength, rand() * roomWidth, rand() * roomHeight];
        distanceAliceToDRIS = norm(alicePos - drisPos);
        distanceDRISToBob = norm(drisPos - bobPos);
        d_alice_bob_i = distanceAliceToDRIS + distanceDRISToBob;
        H_DRIS_iter = H_DRIS_iter + Pt / (d_alice_bob_i^2);
    end
    H_DRIS_total = H_DRIS_total + H_DRIS_iter;
end
H_DRIS = H_DRIS_total / numIterations;

% SNR Calculation
SNR_RIS = (H_RIS^2 * Pt) / sigma_noise;
SNR_DRIS = (H_DRIS^2 * Pt) / sigma_noise;

% Challenge-Response Authentication (Monte Carlo Simulations)
FA_count_DRIS = 0; MD_count_DRIS = 0;
FA_count_RIS = 0; MD_count_RIS = 0;

dictionary = rand(K, M);  % Enrollment phase: random dictionary for authentication

for sim = 1:numSimulations
    % Bob selects a random challenge
    challenge = randi([0, 1]);
    RE_index = randi(M);
    expected_response = dictionary(randi(K), RE_index) * (challenge + 1);
    received_signal_RIS = expected_response + normrnd(0, sqrt(sigma_noise));
    received_signal_DRIS = expected_response + normrnd(0, sqrt(sigma_noise));

    % Eve's attack attempt
    eve_noise = 0.05;
    fake_response = dictionary(randi(K), randi(M)) * (challenge + 1) + normrnd(0, eve_noise);

    % Compute deviations
    deviation_RIS = abs(received_signal_RIS - expected_response);
    deviation_DRIS = abs(received_signal_DRIS - expected_response);
    deviation_Eve = abs(fake_response - expected_response);

    % Adjust FA/MD detection based on spatial diversity impact
    if deviation_RIS > gamma
        MD_count_RIS = MD_count_RIS + 1;
    elseif deviation_Eve < gamma
        FA_count_RIS = FA_count_RIS + 1;
    end

    % DRIS has higher spatial diversity, making Eve’s attack harder
    adjusted_gamma_DRIS = gamma / 2; % More robustness due to distributed REs
    if deviation_DRIS > adjusted_gamma_DRIS
        MD_count_DRIS = MD_count_DRIS + 1;
    elseif deviation_Eve < adjusted_gamma_DRIS * 1.5
        FA_count_DRIS = FA_count_DRIS + 1;
    end
end

% Compute probabilities
FA_prob_RIS = FA_count_RIS / numSimulations;
MD_prob_RIS = MD_count_RIS / numSimulations;
FA_prob_DRIS = FA_count_DRIS / numSimulations;
MD_prob_DRIS = MD_count_DRIS / numSimulations;

% Display results
fprintf('\n--- Comparison of RIS and DRIS ---\n');
fprintf('RIS Channel Gain: %.4f\n', H_RIS);
fprintf('DRIS Channel Gain: %.4f\n', H_DRIS);
fprintf('SNR for RIS: %.2f dB\n', 10 * log10(SNR_RIS));
fprintf('SNR for DRIS: %.2f dB\n', 10 * log10(SNR_DRIS));
fprintf('False Alarm Probability - RIS: %.4f, DRIS: %.4f\n', FA_prob_RIS, FA_prob_DRIS);
fprintf('Misdetection Probability - RIS: %.4f, DRIS: %.4f\n', MD_prob_RIS, MD_prob_DRIS);

%% STEP 17: Weighted Channel Response When Bob is Between 3 DRIS
fprintf('-----------------------------\n');
fprintf("Comparison Between Approximated and Actual Channel Maps in the Real World\n");

% Room dimensions
roomLength = 5; % meters
roomWidth = 5;
roomHeight = 4;

% Bob's Position (x, y, z, rotation)
BobPos = [1.63, 1.59, 1.12, 30];

% Ensure Bob is inside the room
if BobPos(1) < 0 || BobPos(1) > roomLength || BobPos(2) < 0 || BobPos(2) > roomWidth
    error('Bob is outside the room boundaries.');
end
fprintf('Bob Position: (%.2f, %.2f, %.2f, %.2f°)\n', BobPos(1), BobPos(2), BobPos(3), BobPos(4));

% Define DRIS positions in the room
DRIS_positions = [1, 1, 2; 3, 1, 2; 4, 4, 2; 2, 3, 2; 3, 3, 2];
num_DRIS = size(DRIS_positions, 1);

% Compute distances to all DRIS
DRIS_distances = vecnorm(DRIS_positions(:, 1:3) - BobPos(1:3), 2, 2);

% Find the three closest DRIS
[~, sorted_indices] = sort(DRIS_distances);
closest_DRIS = sorted_indices(1:3);

% Compute weights based on inverse distance
weights = 1 ./ (DRIS_distances(closest_DRIS) + 1e-6);
weights = weights / sum(weights);

% Compute weighted channel response
h_DRIS = rand(num_DRIS, 1); % Simulated DRIS channel responses
h_x = sum(weights .* h_DRIS(closest_DRIS));

fprintf('\nEstimated h_total (Weighted DRIS response): %.6f\n', h_x);

%% Step 18: Authentication with Eve's Attack Using Weighted Channel Response for One Position

num_trials = 1000;
threshold = 0.03;
P_FA_count = 0;
P_MD_count = 0;
M = 160;   % Number of RIS configurations
K = 4;    % Number of pilot signals (time slots)
N = 20;    % Number of DRIS elements per configuration

% --- Enrollment Phase ---
dictionary = zeros(K, M * N);  % Stores expected signals for each configuration
for k = 1:K
    for m = 1:(M * N)
        dictionary(k, m) = randn * 0.01 + (rand > 0.5); % Generate pilot signals
    end
end

% Monte Carlo Simulation Loop
for i = 1:num_trials
    % Step 1: Bob sends a random challenge and selects a random RE configuration
    challenge = rand > 0.5;        % Random binary challenge (0 or 1)
    RE_index = randi(M);           % Random RIS element configuration
    RE_elements = (RE_index - 1) * N + (1:N); % Get indices for selected RIS elements
    
    % Step 2: Alice responds based on the enrollment dictionary
    expected_response = h_x * sum(dictionary(randi(K), RE_elements)) / N * (challenge + 1);
    received_signal = expected_response + randn * 0.01 / sqrt(N); % Noise reduction effect
    
    % Step 3: Eve attempts to impersonate Alice
    eve_estimation_noise = 0.05;
    estimated_channel_eve = sum(dictionary(randi(K), randi(M * N, 1, N))) / N + randn * eve_estimation_noise;
    fake_response = estimated_channel_eve * (challenge + 1) + randn * 0.02;
    
    % Compute deviations
    deviation_alice = abs(received_signal - expected_response);
    deviation_eve = abs(fake_response - expected_response);
    
    % Step 4: Apply Heaviside step function correctly
    H_eve = deviation_eve < threshold;   % Eve successfully impersonates Alice (False Alarm)
    H_alice = deviation_alice < threshold; % Alice is correctly authenticated
    
    % Compute probabilities based on correct conditions
    if H_eve == 1  % False Alarm: Eve is mistakenly authenticated
        P_FA_count = P_FA_count + 1;
    end
    if H_alice == 0  % Missed Detection: Alice is mistakenly rejected
        P_MD_count = P_MD_count + 1;
    end
end

% Calculate probabilities
P_FA = P_FA_count / num_trials;
P_MD = P_MD_count / num_trials;

fprintf('\n');
fprintf('----------------------------------------------\n');
fprintf('Probability of False Alarm (PFA): %.4f\n', P_FA);
fprintf('Probability of Missed Detection (PMD): %.4f\n', P_MD);
fprintf('\n');


%% Step 19: Authentication with Eve's Attack Using Weighted Channel Response for Moving User
fprintf('------------------------------------\n');
fprintf("Comparison Between Approximated and Actual Channel Maps in the Real World\n");

% Room dimensions
roomLength = 5; % meters
roomWidth = 5;
roomHeight = 4;

% Initial Bob's Position (x, y, z, rotation)
BobPos = [2.5, 2.5, 1.2, 30]; % Start near center

% Define 30 DRIS positions for better coverage
[X, Y, Z] = meshgrid(linspace(0.5, 4.5, 4), linspace(0.5, 4.5, 4), [1.5, 2.5]);
DRIS_positions = [X(:), Y(:), Z(:)]; % 30 DRIS in a structured grid
num_DRIS = size(DRIS_positions, 1);

% Define a trajectory for Bob to follow (e.g., circular or zig-zag path)
theta = linspace(0, 2*pi, 10); % Circular trajectory
trajectory = [2.5 + 1.2 * cos(theta); 2.5 + 1.2 * sin(theta); ones(1,10) * 1.2]';
num_movements = size(trajectory, 1);

figure(12);clf;
hold on;
xlim([0, roomLength]); ylim([0, roomWidth]); zlim([0, roomHeight]);
view(3);

scatter3(DRIS_positions(:,1), DRIS_positions(:,2), DRIS_positions(:,3), 100, 'b', 'filled');
legend("DRIS Positions", "Location", "best");
total_trials = 1000;
threshold = 0.03;
M = 160;      % Number of RIS configurations
K = 4;     % Number of pilot signals (time slots)
N = 20;     % Number of DRIS elements per configuration

% Generate dictionary
dictionary = zeros(K, M * N);
for k = 1:K
    for m = 1:(M * N)
        dictionary(k, m) = randn * 0.01 + (rand > 0.5);
    end
end

for step = 1:num_movements
    % Update Bob's Position along the trajectory
    BobPos(1:3) = trajectory(step, :);
    
    % Compute distances to all DRIS
    DRIS_distances = vecnorm(DRIS_positions(:, 1:3) - BobPos(1:3), 2, 2);
    
    % Select all DRIS within a fixed radius (e.g., 2 meters) for better accuracy
    radius = 2;
    close_DRIS = find(DRIS_distances < radius);
    
    % If fewer than 3 DRIS are selected, take the nearest ones
    if length(close_DRIS) < 3
        [~, sorted_indices] = sort(DRIS_distances);
        close_DRIS = sorted_indices(1:3);
    end
    
    % Compute Gaussian-based weights
    sigma = 0.5;
    weights = exp(-DRIS_distances(close_DRIS).^2 / (2 * sigma^2));
    weights = weights / sum(weights);
    
    % Compute weighted channel response
    h_DRIS = rand(num_DRIS, 1); % Simulated DRIS channel responses
    h_x = sum(weights .* h_DRIS(close_DRIS));
    
    fprintf('Step %d - Bob Position: (%.2f, %.2f, %.2f)\n', step, BobPos(1), BobPos(2), BobPos(3));
    
    % Reset false alarm and missed detection counts for each step
    P_FA_count = 0;
    P_MD_count = 0;
    
    % Authentication process
    for i = 1:total_trials
        challenge = rand > 0.5;
        RE_index = randi(M);
        RE_elements = (RE_index - 1) * N + (1:N); % Select indices for RIS elements
        
        % Alice's expected response using weighted RIS elements
        expected_response = h_x * sum(dictionary(randi(K), RE_elements)) / N * (challenge + 1);
        received_signal = expected_response + randn * 0.01 / sqrt(N); % Noise reduction
        
        % Eve's attack using random RIS elements
        eve_estimation_noise = 0.05;
        estimated_channel_eve = sum(dictionary(randi(K), randi(M * N, 1, N))) / N + randn * eve_estimation_noise;
        fake_response = estimated_channel_eve * (challenge + 1) + randn * 0.02;
        
        % Compute deviations
        deviation_alice = abs(received_signal - expected_response);
        deviation_eve = abs(fake_response - expected_response);
        
        % Decision rules
        H_eve = deviation_eve < threshold;   % Eve successfully impersonates Alice (False Alarm)
        H_alice = deviation_alice < threshold; % Alice is correctly authenticated
        
        % Count authentication errors
        if H_eve == 1  % False Alarm: Eve is mistakenly authenticated
            P_FA_count = P_FA_count + 1;
        end
        if H_alice == 0  % Missed Detection: Alice is mistakenly rejected
            P_MD_count = P_MD_count + 1;
        end
    end
    
    % Compute probabilities
    P_FA = P_FA_count / total_trials;
    P_MD = P_MD_count / total_trials;
    
    fprintf('Step %d - P_FA: %.4f, P_MD: %.4f\n', step, P_FA, P_MD);
    
    figure(12);
    % Visualization of Bob's movement
    scatter3(BobPos(1), BobPos(2), BobPos(3), 150, 'r', 'filled');
    pause(0.5);
end

hold off;
%%
