% 灰度直方图显示和均衡化
function showHistogramAndEqualize(app, srcImage)
    grayImage = rgb2gray(srcImage);
    figure;
    subplot(1,2,1);
    imhist(grayImage);
    title('Original Histogram');
    
    equalizedImage = histeq(grayImage);
    subplot(1,2,2);
    imhist(equalizedImage);
    title('Equalized Histogram');
end

% 对比度增强 - 线性变换
function enhanceContrastLinear(app, srcImage)
    % 线性变换公式：f(x) = alpha * x + beta
    alpha = 1.5; % 增强因子
    beta = 0; % 偏移量
    enhancedImage = imadjust(srcImage, [], [], alpha, beta);
    imshow(enhancedImage);
end

% 对比度增强 - 对数变换
function enhanceContrastLog(app, srcImage)
    c = 1; % 对数变换常数
    enhancedImage = log(double(srcImage) + 1) / log(c);
    imshow(enhancedImage, []);
end

% 图像缩放
function scaleImage(app, srcImage)
    scale = 0.5; % 缩放因子
    scaledImage = imresize(srcImage, scale);
    imshow(scaledImage);
end

% 图像旋转
function rotateImage(app, srcImage)
    angle = 45; % 旋转角度
    rotatedImage = imrotate(srcImage, angle);
    imshow(rotatedImage);
end

% 图像加噪和滤波
function addNoiseAndFilter(app, srcImage)
    noiseImage = imnoise(srcImage, 'gaussian', 0, 0.01);
    imshow(noiseImage);
    
    % 空域滤波
    filteredImage = imgaussfilt(noiseImage, 2);
    imshow(filteredImage);
    
    % 频域滤波
    F = fft2(double(noiseImage));
    H = fspecial('gaussian', size(F), 10);
    G = H .* F;
    filteredImageF = ifft2(G);
    imshow(filteredImageF, []);
end

% 边缘提取
function edgeDetection(app, srcImage)
    edgesRobert = edge(srcImage, 'roberts');
    edgesPrewitt = edge(srcImage, 'prewitt');
    edgesSobel = edge(srcImage, 'sobel');
    edgesLaplacian = edge(srcImage, 'log');
    
    figure;
    subplot(2,2,1), imshow(edgesRobert), title('Robert');
    subplot(2,2,2), imshow(edgesPrewitt), title('Prewitt');
    subplot(2,2,3), imshow(edgesSobel), title('Sobel');
    subplot(2,2,4), imshow(edgesLaplacian), title('Laplacian');
end

% 特征提取 - LBP
function extractLBPFeatures(app, srcImage)
    % 这里需要实现LBP特征提取的代码
end

% 特征提取 - HOG
function extractHOGFeatures(app, srcImage)
    % 这里需要实现HOG特征提取的代码
end

% 图像分类（加分项）
function classifyImage(app, srcImage)
    % 这里需要实现图像分类的代码，可以使用传统机器学习方法或深度学习
end