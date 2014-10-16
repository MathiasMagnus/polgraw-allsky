function  [M,fftpad,gamrn,Mn] = freadGrid(GridFile,DataDir)
% FREADGRID 
% Struktura plik�w jest nast�puj�ca.  pierwsze 4 bajty: fftpad (typ 
% integer), nast�pne 8x16 bajt�w: macierz M u�o�ona wierszami (tablica 
% liczb double), nast�pne 8x16 bajt�w: macierz gamrn, nast�pne 8x16: 
% macierz Mn.
% grid_01.bin    MM=sqrt(3/4)      \theta = 2.618
% grid_02.bin    MM=0.9            \theta = 2.282
% grid_03.bin    MM=0.9^(1/3)      \theta = 2.047
% Examples:
% M = freadGrid('grid.bin','D:\Gauss1\03');
% [M,fftpad,gamrn,Mn] = freadGrid('grid.bin','D:\VirgoData\AllSkySearch\VSR1\01');  
%

hdr = pwd;
cd(DataDir)
fid = fopen(GridFile,'r');
fftpad = fread(fid,1,'int');
M = fread(fid,[4,4],'double');
gamrn = fread(fid,[4,4],'double');
Mn = fread(fid,[4,4],'double');
fclose(fid);
M = M';
Mn = Mn';
cd(hdr)
