% save jpg images as bin file for cpp
%
is_server = 0;

dataset = 'coco';  %'coco', 'voc2012'

if is_server
  if strcmp(dataset, 'voc2012')
    img_folder  = '/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages'
    save_folder = '/rmt/data/pascal/VOCdevkit/VOC2012/PPMImages';
  elseif strcmp(dataset, 'coco')
    img_folder  = '/rmt/data/coco/JPEGImages';
    save_folder = '/rmt/data/coco/PPMImages';
  end
else
%  img_folder = '../img';
%  save_folder = '../img_ppm';
  img_folder = '/home/rasha/Desktop/example';
  save_folder = '/home/rasha/Desktop/example';
end

if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

img_dir = dir(fullfile(img_folder, '*-Input.tif'));

for i = 1 : numel(img_dir)
    fprintf(1, 'processing %d (%d)...\n', i, numel(img_dir));
    img = imread(fullfile(img_folder, img_dir(i).name));
    
    img_fn = img_dir(i).name(1:end-4);
    save_fn = fullfile(save_folder, [img_fn, '.ppm']);
    
    imwrite(img, save_fn);   
end
    
