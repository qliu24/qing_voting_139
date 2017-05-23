for d = 40: nd
    % display progress
    if toc > 1
        fprintf('%s: pr: compute: %d|%d\n', category, d, nd);
        drawnow;
        tic;
    end
    
    % find ground truth image
    i = img_ids(d);   % i: image id
    
    if i == 0
        fp(d) = 1;
    else
        
        % assign detection to ground truth object if any
        bb = boxes(:, d);
        ovmax = -inf;
        jmax = [];
        for j = 1: size(gt(i).bbox, 2)
            bbgt = gt(i).bbox(:, j);
            bi = [max(bb(1), bbgt(1)); max(bb(2), bbgt(2)); min(bb(3), bbgt(3)); min(bb(4), bbgt(4))];   % intersection
            iw = bi(3) - bi(1) + 1;
            ih = bi(4) - bi(2) + 1;
            if (iw > 0) && (ih > 0)                
                % compute overlap as area of intersection / area of union
                ua = (bb(3) - bb(1) + 1) * (bb(4) - bb(2) + 1) + ...
                     (bbgt(3) - bbgt(1) + 1) * (bbgt(4) - bbgt(2) + 1) - ...
                      iw * ih;    % area of union
                ov = iw * ih / ua;
                if ov > ovmax
                    ovmax = ov;
                    jmax = j;
                end
            end
        end
        
        % assign detection as true positive/don't care/false positive
        if ovmax >= Eval.ov_thresh
            if ~gt(i).diff(jmax)
                if ~gt(i).det(jmax)
                    tp(d) = 1;            % true positive
                    gt(i).det(jmax) = true;
                else
                    fp(d) = 1;            % false positive (multiple/duplicate detection)
                end
            end
        else
            fp(d)=1;                    % false positive
        end
    end % if i == 0
    
    if fp(d) == 1
        d
        break
    end
end

n=img_ids(d);
bbox = boxes(:,d);
img_name = img_list{1}{n};
file_img = sprintf('%s/%s.JPEG', dir_img, img_name);
img = imread(file_img);
img1 = img(bbox(2):bbox(4), bbox(1):bbox(3));

gtbox=gt(n).bbox;
img2 = img(gtbox(2):gtbox(4), gtbox(1):gtbox(3));
imshowpair(img1, img2, 'montage');

ov=0;
bi = [max(bbox(1), gtbox(1)); max(bbox(2), gtbox(2)); min(bbox(3), gtbox(3)); min(bbox(4), gtbox(4))];   % intersection
iw = bi(3) - bi(1) + 1;
ih = bi(4) - bi(2) + 1;
if (iw > 0) && (ih > 0)                
    ua = (bbox(3) - bbox(1) + 1) * (bbox(4) - bbox(2) + 1) + ...
                     (gtbox(3) - gtbox(1) + 1) * (gtbox(4) - gtbox(2) + 1) - ...
                      iw * ih;    % area of union
    ov = iw * ih / ua;
end
ov