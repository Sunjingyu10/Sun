function image_processing_gui()
    % 创建主界面
    fig = uifigure('Position', [100, 100, 800, 600], 'Name', '图像处理工具');

    % 创建菜单
    menu = uimenu(fig, 'Text', '文件');
    uimenu(menu, 'Text', '打开图像', 'MenuSelectedFcn', @(src, event) open_image(fig));
    
    % 创建按钮
    uibutton(fig, 'Text', '直方图均衡化', 'Position', [50, 500, 150, 30], ...
        'ButtonPushedFcn', @(src, event) histogram_equalization(fig));
    uibutton(fig, 'Text', '对比度增强', 'Position', [250, 500, 150, 30], ...
        'ButtonPushedFcn', @(src, event) enhance_contrast(fig));
    uibutton(fig, 'Text', '图像变换', 'Position', [450, 500, 150, 30], ...
        'ButtonPushedFcn', @(src, event) image_transformations(fig));
    uibutton(fig, 'Text', '噪声与滤波', 'Position', [650, 500, 150, 30], ...
        'ButtonPushedFcn', @(src, event) noise_and_filtering(fig));
    uibutton(fig, 'Text', '边缘提取', 'Position', [50, 450, 150, 30], ...
        'ButtonPushedFcn', @(src, event) edge_detection(fig));
    uibutton(fig, 'Text', '特征提取', 'Position', [250, 450, 150, 30], ...
        'ButtonPushedFcn', @(src, event) feature_extraction(fig));
end

function open_image(fig)
    % 直接加载 hudie.png
    imagePath = 'hudie.png'; % 确保此路径正确
    if exist(imagePath, 'file') ~= 2
        uialert(fig, '图像文件 hudie.png 未找到！', '错误');
        return;
    end
    
    img = imread(imagePath);
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    
    % 显示图像
    ax = uiaxes(fig, 'Position', [50, 50, 700, 350]);
    imshow(grayImg, 'Parent', ax);
    title(ax, '原始图像');
    
    % 存储灰度图像
    fig.UserData.grayImg = grayImg; 
    disp('图像已加载。'); % 调试信息
end

function histogram_equalization(fig)
    if ~isfield(fig.UserData, 'grayImg')
        uialert(fig, '请先加载图像！', '错误');
        return;
    end
    grayImg = fig.UserData.grayImg;
    eqImg = histeq(grayImg);
    
    % 显示结果
    figure;
    subplot(1, 2, 1);
    imhist(grayImg);
    title('原始直方图');
    
    subplot(1, 2, 2);
    imhist(eqImg);
    title('均衡化直方图');
    
    figure;
    imshow(eqImg);
    title('均衡化图像');
end

function enhance_contrast(fig)
    if ~isfield(fig.UserData, 'grayImg')
        uialert(fig, '请先加载图像！', '错误');
        return;
    end
    grayImg = fig.UserData.grayImg;
    
    % 线性变换
    alpha = 1.5; % 增强因子
    linearEnhanced = imadjust(grayImg, [], [], alpha);
    
    % 对数变换
    logEnhanced = log(double(grayImg) + 1);
    logEnhanced = im2uint8(logEnhanced / max(logEnhanced(:)));
    
    % 指数变换
    gamma = 0.5;
    expEnhanced = im2uint8((double(grayImg) / 255) .^ gamma * 255);
    
    % 显示结果
    figure;
    subplot(1, 3, 1);
    imshow(linearEnhanced);
    title('线性对比度增强');
    
    subplot(1, 3, 2);
    imshow(logEnhanced);
    title('对数对比度增强');
    
    subplot(1, 3, 3);
    imshow(expEnhanced);
    title('指数对比度增强');
end

function image_transformations(fig)
    if ~isfield(fig.UserData, 'grayImg')
        uialert(fig, '请先加载图像！', '错误');
        return;
    end
    grayImg = fig.UserData.grayImg;
    
    % 图像缩放
    scaledImg = imresize(grayImg, 0.5);
    
    % 图像旋转
    rotatedImg = imrotate(grayImg, 45);
    
    % 显示结果
    figure;
    subplot(1, 2, 1);
    imshow(scaledImg);
    title('缩放图像');
    
    subplot(1, 2, 2);
    imshow(rotatedImg);
    title('旋转图像');
end

function noise_and_filtering(fig)
    if ~isfield(fig.UserData, 'grayImg')
        uialert(fig, '请先加载图像！', '错误');
        return;
    end
    grayImg = fig.UserData.grayImg;
    
    % 加噪声
    noisyImg = imnoise(grayImg, 'gaussian', 0, 0.01);
    
    % 空域滤波
    gaussFilteredImg = imgaussfilt(noisyImg, 2);
    
    % 频域滤波
    F = fft2(double(noisyImg));
    H = fspecial('gaussian', size(F), 10);
    G = H .* F;
    freqFilteredImg = ifft2(G);
    freqFilteredImg = real(freqFilteredImg);
    
    % 显示结果
    figure;
    subplot(1, 3, 1);
    imshow(noisyImg);
    title('加噪声图像');
    
    subplot(1, 3, 2);
    imshow(gaussFilteredImg);
    title('高斯滤波');
    
    subplot(1, 3, 3);
    imshow(freqFilteredImg, []);
    title('频域滤波');
end

function edge_detection(fig)
    if ~isfield(fig.UserData, 'grayImg')
        uialert(fig, '请先加载图像！', '错误');
        return;
    end
    grayImg = fig.UserData.grayImg;
    
    % 使用不同算子进行边缘检测
    robertEdges = edge(grayImg, 'roberts');
    prewittEdges = edge(grayImg, 'prewitt');
    sobelEdges = edge(grayImg, 'sobel');
    laplacianEdges = edge(grayImg, 'log');
    
    % 显示结果
    figure;
    subplot(2, 2, 1);
    imshow(robertEdges);
    title('Robert算子');
    
    subplot(2, 2, 2);
    imshow(prewittEdges);
    title('Prewitt算子');
    
    subplot(2, 2, 3);
    imshow(sobelEdges);
    title('Sobel算子');
    
    subplot(2, 2, 4);
    imshow(laplacianEdges);
    title('拉普拉斯算子');
end

function feature_extraction(fig)
    if ~isfield(fig.UserData, 'grayImg')
        uialert(fig, '请先加载图像！', '错误');
        return;
    end
    grayImg = fig.UserData.grayImg;
    
    % LBP特征提取
    lbpFeatures = extractLBPFeatures(grayImg);
    
    % HOG特征提取
    [hogFeatures, ~] = extractHOGFeatures(grayImg);
    
    % 显示结果
    disp('LBP特征:');
    disp(lbpFeatures);
    
    disp('HOG特征:');
    disp(hogFeatures);
end