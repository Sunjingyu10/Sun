function histogram_operations()
    % 读取图像
    imagePath = 'hudie.png';
    img = imread(imagePath);
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end

    % 显示原始灰度直方图
    figure;
    subplot(1,3,1);
    imhist(grayImg);
    title('Original Histogram');

    % 直方图均衡化
    eqImg = histeq(grayImg);
    subplot(1,3,2);
    imhist(eqImg);
    title('Equalized Histogram');
    imshow(eqImg), title('Equalized Image');

    % 直方图匹配（规定化）
    targetHist = ones(1, 256) / 256; % 均匀分布的目标直方图
    matchedImg = histeq(grayImg); % 先均衡化
    for i = 1:256
        % 找到累积直方图
        [~, idx] = min(abs(cdf(eqImg) - i/256));
        matchedImg(grayImg == i) = idx;
    end
    subplot(1,3,3);
    imhist(matchedImg);
    title('Matched Histogram');
    imshow(matchedImg), title('Matched Image');
end