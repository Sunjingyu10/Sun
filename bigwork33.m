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
    uibutton(fig, 'Text', '目标提取', 'Position', [450, 450, 150, 30], ...
        'ButtonPushedFcn', @(src, event) target_extraction(fig));
end

function open_image(fig)
    % 选择图像文件
    [file, path] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp', '图像文件 (*.png, *.jpg, *.jpeg, *.bmp)'}, '选择图像文件');
    if isequal(file, 0)
        return; % 用户取消选择
    end
    
    imagePath = fullfile(path, file);
    img = imread(imagePath);
    
    % 显示原图和灰度图
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    
    % 显示图像
    ax1 = uiaxes(fig, 'Position', [50, 50, 300, 300]);
    imshow(img, 'Parent', ax1);
    title(ax1, '原始图像');
    
    ax2 = uiaxes(fig, 'Position', [400, 50, 300, 300]);
    imshow(grayImg, 'Parent', ax2);
    title(ax2, '灰度图像');
    
    % 存储灰度图像和原始图像
    fig.UserData.grayImg = grayImg; 
    fig.UserData.originalImg = img; % 存储原始图像
    disp('图像已加载。'); % 调试信息
end

function target_extraction(fig)
    if ~isfield(fig.UserData, 'grayImg')
        uialert(fig, '请先加载图像！', '错误');
        return;
    end
    grayImg = fig.UserData.grayImg;
    
    % 进行二值化处理（自适应阈值）
    binary_image = imbinarize(grayImg, 'adaptive');

    % 定义击中部分的结构元素
    se_hit = strel('disk', 3);

    % 定义未击中部分的结构元素
    disk_matrix = zeros(5); % 创建一个5x5的全0矩阵
    [rows, cols] = find(imerode(ones(5), strel('disk', 2))); % 找到内径为2的圆形区域位置
    disk_matrix(sub2ind(size(disk_matrix), rows, cols)) = 1; % 将环形区域对应的位置设为1
    se_miss = strel('arbitrary', disk_matrix); % 用自定义矩阵初始化结构元素

    % 对图像进行击中部分的腐蚀操作
    eroded_hit = imerode(binary_image, se_hit);

    % 对图像取反
    complemented_image = ~binary_image;

    % 对取反后的图像进行未击中部分的腐蚀操作
    eroded_miss = imerode(complemented_image, se_miss);

    % 再将腐蚀后的背景取反回来
    eroded_miss = ~eroded_miss;

    % 执行逻辑与操作，得到击中与否变换的结果
    result = eroded_hit & eroded_miss;

    % 显示结果
    figure;
    subplot(2, 3, 1);
    imshow(grayImg);
    title('灰度图像');

    subplot(2, 3, 2);
    imshow(result);
    title('目标提取结果');

    % 提取原始图像的LBP特征
    [N, M] = size(grayImg);
    lbp = zeros(N, M);
    for j = 2:(N - 1)
        for i = 2:(M - 1)
            neighbor = [grayImg(j - 1, i - 1), grayImg(j - 1, i), grayImg(j - 1, i + 1);
                        grayImg(j, i - 1), grayImg(j, i), grayImg(j, i + 1);
                        grayImg(j + 1, i - 1), grayImg(j + 1, i), grayImg(j + 1, i + 1)];
            count = 0;
            for k = 0:7
                binVal1 = neighbor(mod(k + 1, 3) + 1, ceil((k + 1) / 3));
                binVal2 = grayImg(j, i);
                if binVal1 > binVal2
                    count = count + 2 ^ (7 - k);
                end
            end
            lbp(j, i) = count;
        end
    end
    lbp = uint8(lbp);
    disp('原始图像的LBP特征：');
    disp(lbp);

    % 绘制原始图像的LBP特征图像
    original_lbp_image = mat2gray(lbp); 
    subplot(2, 3, 3);
    imshow(original_lbp_image);
    title('原始图像的LBP特征图像');

    % 提取原始图像的HOG特征
    [original_hog_feat, visualization] = extractHOGFeatures(grayImg);
    disp('原始图像的HOG特征：');
    disp(original_hog_feat);

    % 绘制原始图像的HOG特征图像
    if isstruct(visualization) && isfield(visualization, 'Visualization')
        hog_image_data = visualization.Visualization; 
        subplot(2, 3, 4);
        imshow(hog_image_data);
        title('原始图像 HOG特征图像');
    else
        warning('无法正确解析HOG特征可视化数据，可能版本不兼容');
    end

    % 提取目标图像的LBP特征
    [N, M] = size(result);
    lbp_target = zeros(N, M);
    for j = 2:(N - 1)
        for i = 2:(M - 1)
            neighbor = [result(j - 1, i - 1), result(j - 1, i), result(j - 1, i + 1);
                        result(j, i - 1), result(j, i), result(j, i + 1);
                        result(j + 1, i - 1), result(j + 1, i), result(j + 1, i + 1)];
            count = 0;
            for k = 0:7
                binVal1 = neighbor(mod(k + 1, 3) + 1, ceil((k + 1) / 3));
                binVal2 = result(j, i);
                if binVal1 > binVal2
                    count = count + 2 ^ (7 - k);
                end
            end
            lbp_target(j, i) = count;
        end
    end
    lbp_target = uint8(lbp_target);
    disp('提取目标图像的LBP特征：');
    disp(lbp_target);

    % 绘制提取目标图像的LBP特征图像
    target_lbp_image = mat2gray(lbp_target);
    subplot(2, 3, 5);
    imshow(target_lbp_image);
    title('目标图像的LBP特征图像');

    % 提取目标图像的HOG特征
    [target_hog_feat, visualization_target] = extractHOGFeatures(result);
    disp('提取目标图像的HOG特征：');
    disp(target_hog_feat);

    % 绘制提取目标图像的HOG特征图像
    if isstruct(visualization_target) && isfield(visualization_target, 'Visualization')
        hog_image_data_target = visualization_target.Visualization;
        subplot(2, 3, 6);
        imshow(hog_image_data_target);
        title('目标图像 HOG特征图像');
    else
        warning('无法正确解析HOG特征可视化数据，可能版本不兼容');
    end
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
    if ~isfield(fig.UserData, 'grayImg') || ~isfield(fig.UserData, 'targetImg')
        uialert(fig, '请先加载图像和提取目标！', '错误');
        return;
    end
    grayImg = fig.UserData.grayImg;
    targetImg = fig.UserData.targetImg;
    
    % 对原始图像进行特征提取
    disp('原始图像特征提取:');
    lbpFeaturesOriginal = extractLBPFeatures(grayImg);
    [hogFeaturesOriginal, hogVisualizationOriginal] = extractHOGFeatures(grayImg, 'CellSize', [32 32]);
    
    % 对提取的目标进行特征提取
    disp('提取目标特征提取:');
    lbpFeaturesTarget = extractLBPFeatures(targetImg);
    [hogFeaturesTarget, hogVisualizationTarget] = extractHOGFeatures(targetImg, 'CellSize', [32 32]);
    
    % 显示结果
    disp('原始图像 LBP特征:');
    disp(lbpFeaturesOriginal);
    
    disp('原始图像 HOG特征:');
    disp(hogFeaturesOriginal);
    
    disp('提取目标 LBP特征:');
    disp(lbpFeaturesTarget);
    
    disp('提取目标 HOG特征:');
    disp(hogFeaturesTarget);
    
    % 可视化原始图像的HOG特征
    figure;
    imshow(hogVisualizationOriginal);
    title('原始图像 HOG特征图像');
    
    % 可视化提取目标的HOG特征
    figure;
    imshow(hogVisualizationTarget);
    title('提取目标 HOG特征图像');
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