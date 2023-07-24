function [] = add_noise(noise_file,path_clean_file,extension,SNR,saving_path)
addpath(genpath(path_clean_file))
cleanFiles = dir([path_clean_file,'*.',extension]);
for i=1:length(cleanFiles)
    
            disp(['adding noise ',noise_file,' to file ',cleanFiles(i).name,' at SNR ',mat2str(SNR),'...'])
            [y1Mono,fs] = ster2mono(cleanFiles(i).name);
            %[x1,fs]=audioread(cleanFiles(i).name)
            [y2Mono] = ster2mono(noise_file);
            %[y2Mono,fs2]=audioread(noise_file)
            %function [y,yMono,fs] = ster2mono(path,filename)
            noisy_sig=sigmerge(y1Mono,y2Mono(1:length(y1Mono)),SNR);
            
            disp('saving the noisy file...')
            noisyFile=['noisy',cleanFiles(i).name(1:end-4),'_SNR',mat2str(SNR),'_noise',noise_file(1:end-4),'.wav'];
            %audiowrite([path,'\',filename(1:end-4),'mono.wav'],yMono,fs)
            audiowrite([saving_path,'\',noisyFile],noisy_sig,fs)

         
end

