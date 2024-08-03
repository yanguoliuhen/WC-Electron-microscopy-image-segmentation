import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WC稀疏图像粒度分析软件 v1.0")
        self.root.geometry("2200x1000")
        self.root.resizable(False, False)
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12), padding=10)
        style.configure("TLabel", font=("Helvetica", 12), padding=10)
        style.configure("TFrame", background="#f0f0f0")
        self.frame_src = ttk.Frame(root, borderwidth=2, relief=tk.GROOVE)
        self.frame_src.grid(row=0, column=0, padx=20, pady=20, sticky="n")
        self.frame_des = ttk.Frame(root, borderwidth=2, relief=tk.GROOVE)
        self.frame_des.grid(row=0, column=1, padx=20, pady=20, sticky="n")
        self.canvas_src = tk.Canvas(self.frame_src, bg="white", width=1024, height=768)
        self.canvas_src.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar_src = ttk.Scrollbar(self.frame_src, orient=tk.VERTICAL, command=self.canvas_src.yview)
        self.scrollbar_src.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_src.configure(yscrollcommand=self.scrollbar_src.set)
        self.canvas_src.bind('<Configure>', lambda e: self.canvas_src.configure(scrollregion=self.canvas_src.bbox("all")))
        self.canvas_des = tk.Canvas(self.frame_des, bg="white", width=1024, height=768)
        self.canvas_des.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar_des = ttk.Scrollbar(self.frame_des, orient=tk.VERTICAL, command=self.canvas_des.yview)
        self.scrollbar_des.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_des.configure(yscrollcommand=self.scrollbar_des.set)
        self.canvas_des.bind('<Configure>', lambda e: self.canvas_des.configure(scrollregion=self.canvas_des.bbox("all")))
        self.label_src_image = None
        self.label_des_image = None
        self.path = ''
        self.image = None
        btn_frame = ttk.Frame(root)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=20)
        btn_load = ttk.Button(btn_frame, text="导入图片", command=self.load_image)
        btn_load.grid(row=0, column=0, padx=10)
        btn_segment = ttk.Button(btn_frame, text="分析图片", command=self.segment_image)
        btn_segment.grid(row=0, column=1, padx=10)
        btn_clear = ttk.Button(btn_frame, text="清除图片", command=self.clear_image)
        btn_clear.grid(row=0, column=2, padx=10)
        btn_save = ttk.Button(btn_frame, text="保存结果", command=self.save_results)
        btn_save.grid(row=0, column=3, padx=10)
        lbl_threshold = ttk.Label(btn_frame, text="输入阈值：", font=("Helvetica", 12))
        lbl_threshold.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.threshold_entry = tk.Entry(btn_frame, font=("Helvetica", 12), width=40)
        self.threshold_entry.grid(row=1, column=1, columnspan=3, padx=10, pady=10, sticky="w")
        self.threshold_entry.insert(0, "0.28, 0.19, 0.21, 0.41, 0.51, 0.52, 0.597")
    def load_image(self):
        self.path = filedialog.askopenfilename()
        if self.path:
            self.image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)
            b, g, r = cv2.split(self.image)
            self.image = cv2.merge([r, g, b])
            image_pil = Image.fromarray(self.image)
            tk_image = ImageTk.PhotoImage(image_pil)
            self.canvas_src.create_image(0, 0, image=tk_image, anchor=tk.NW)
            self.canvas_src.image = tk_image
            self.canvas_src.configure(scrollregion=self.canvas_src.bbox("all"))
    def segment_image(self):
        if self.image is None:
            return
        result_image, diameter_image = self.watershed_algorithm(self.image)
        result_pil = Image.fromarray(result_image)
        tk_result_image = ImageTk.PhotoImage(result_pil)
        self.canvas_des.create_image(0, 0, image=tk_result_image, anchor=tk.NW)
        self.canvas_des.image = tk_result_image
        self.canvas_des.configure(scrollregion=self.canvas_des.bbox("all"))
    def clear_image(self):
        self.canvas_src.delete("all")
        self.canvas_des.delete("all")
        self.image = None
    def merge_regions(self, contours, threshold_distance, min_area):
        merged_regions = {}
        for i in range(len(contours)):
            contour_i = contours[i]
            merged = False
            area_i = cv2.contourArea(contour_i)
            if area_i < min_area:
                continue
            for j in range(i+1, len(contours)):
                contour_j = contours[j]
                distance = cv2.matchShapes(contour_i, contour_j, cv2.CONTOURS_MATCH_I1, 0)
                area_j = cv2.contourArea(contour_j)
                if area_i < area_j and distance < threshold_distance:
                    merged_regions[j] = merged_regions.get(j, []) + [contour_i]
                    merged = True
                elif area_i > area_j and distance < threshold_distance:
                    merged_regions[i] = merged_regions.get(i, []) + [contour_j]
                    merged = True
            if not merged:
                merged_regions[i] = merged_regions.get(i, []) + [contour_i]
        return merged_regions
    def remove_overlapping_contours(self, contours):
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
        contours_to_remove = []
        for i in range(len(bounding_boxes)):
            for j in range(i+1, len(bounding_boxes)):
                box_i = bounding_boxes[i]
                box_j = bounding_boxes[j]
                intersection_x = max(box_i[0], box_j[0])
                intersection_y = max(box_i[1], box_j[1])
                intersection_w = min(box_i[0]+box_i[2], box_j[0]+box_j[2]) - intersection_x
                intersection_h = min(box_i[1]+box_i[3], box_j[1]+box_j[3]) - intersection_y
                if intersection_w > 20 and intersection_h > 20:
                    contours_to_remove.append(i)
                    break
        cleaned_contours = [cnt for i, cnt in enumerate(contours) if i not in contours_to_remove]
        return cleaned_contours
    def watershed_algorithm(self, image):
        try:
            threshold_values = list(map(float, self.threshold_entry.get().split(',')))
        except ValueError:
            threshold_values = [0.28, 0.19, 0.21, 0.41, 0.51, 0.52, 0.597]
        src = image.copy()
        blur = cv2.pyrMeanShiftFiltering(image, sp=21, sr=55)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        smoothed_image = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
        segmented_images = []
        for threshold in threshold_values:
            _, surface = cv2.threshold(dist_out, threshold * dist_out.max(), 255, cv2.THRESH_BINARY)
            sure_fg = np.uint8(surface)
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=10)
            unknown = binary - sure_fg
            markers[unknown == 255] = 0
            markers = cv2.watershed(image, markers=markers)
            segmented_images.append(markers)
        combined_markers = np.zeros_like(segmented_images[0])
        for markers in segmented_images:
            combined_markers += markers
        min_val, max_val, _, _ = cv2.minMaxLoc(combined_markers)
        markers_8u = np.uint8(combined_markers)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                  (128, 128, 128), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128),
                  (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0), (128, 0, 255), (255, 128, 128),
                  (128, 255, 255)]
        contours = []
        for i in range(2, int(max_val + 1)):
            if i != len(threshold_values):
                ret, thres1 = cv2.threshold(markers_8u, i - 1, 255, cv2.THRESH_BINARY)
                ret2, thres2 = cv2.threshold(markers_8u, i, 255, cv2.THRESH_BINARY)
                mask = thres1 - thres2
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours.extend(cnts)
        merged_regions = self.merge_regions(contours, 0, 50)
        all_contours = []
        for contours in merged_regions.values():
            all_contours.extend(contours)
        merged_regions = self.remove_overlapping_contours(all_contours)
        non_overlapping_regions = {}
        for i, contour in enumerate(merged_regions):
            non_overlapping_regions[i] = [contour]
        height, width = image.shape[:2]
        black_background = np.zeros((height, width, 3), dtype=np.uint8)
        for i, contours in non_overlapping_regions.items():
            color = colors[i % len(colors)]
            for contour in contours:
                cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
        result = cv2.addWeighted(image, 0.5, src, 0.5, 0)
        for i, contours in non_overlapping_regions.items():
            color = colors[i % len(colors)]
            for contour in contours:
                cv2.drawContours(black_background, [contour], -1, color, thickness=cv2.FILLED)
        result = cv2.addWeighted(black_background, 0.5, src, 0.5, 0)
        equivalent_diameters = []
        for i, contours in non_overlapping_regions.items():
            for contour in contours:
                area = cv2.contourArea(contour)
                equivalent_diameter = 2 * np.sqrt(area / np.pi) * 0.02324219
                equivalent_diameters.append(equivalent_diameter)
        for i, contours in non_overlapping_regions.items():
            color = colors[i % len(colors)]
            for contour in contours:
                area = cv2.contourArea(contour)
                equivalent_diameter = 2 * np.sqrt(area / np.pi) * 0.02324219
                if 0 <= equivalent_diameter <= 0.48:
                    (x, y), _ = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    cv2.putText(black_background, f'{equivalent_diameter:.2f}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        result_diameters = cv2.addWeighted(black_background, 0.5, src, 0.5, 0)
        bins = [0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.8]
        plt.hist(equivalent_diameters, bins=bins, edgecolor='black')
        plt.xlabel('Equivalent Diameter')
        plt.ylabel('Count')
        plt.title('Particle Size Distribution')
        plt.xticks(bins)
        plt.show()
        return result, result_diameters
    def save_results(self):
        if self.image is None:
            return
        result_image, diameter_image = self.watershed_algorithm(self.image)
        result_pil = Image.fromarray(result_image)
        result_pil.save('segmented_image.png')
        equivalent_diameters = []
        for i, contours in self.non_overlapping_regions.items():
            for contour in contours:
                area = cv2.contourArea(contour)
                equivalent_diameter = 2 * np.sqrt(area / np.pi) * 0.02324219
                equivalent_diameters.append(equivalent_diameter)
        with open('equivalent_diameters.txt', 'w') as f:
            f.write('Equivalent Diameters:\n')
            for diam in equivalent_diameters:
                f.write(f'{diam:.2f}\n')
        print('Results saved successfully.')
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()