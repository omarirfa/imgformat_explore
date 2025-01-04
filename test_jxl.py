import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import time
    from datetime import datetime
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.ndimage import convolve, gaussian_filter
    import polars as pl
    from io import BytesIO
    import colorsys
    from matplotlib.colors import hsv_to_rgb
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    from skimage.metrics import structural_similarity as ssim
    from skimage.util import compare_images
    from skimage import exposure
    # import pillow_jxl
    # from PIL import Image
    # from IPython.display import display

    class ImageFormatAnalyzer:
        def __init__(
            self, input_path, output_dir="output_imgs", use_parallel=True, max_workers=8
        ):
            """Initialize the analyzer with an input image path."""
            self.input_path = input_path
            self.output_dir = output_dir
            self.use_parallel = use_parallel
            self.max_workers = max_workers

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            print(f"\nAnalyzing input image: {os.path.basename(input_path)}")
            print(f"Input format: {os.path.splitext(input_path)[1].upper()}")
            print(f"Output directory: {output_dir}")

            print("Loading original image...")
            with Image.open(input_path) as img:
                # Ensure minimum size requirements are met
                width, height = img.size
                if width < 7 or height < 7:
                    raise ValueError("Image dimensions must be at least 7x7 pixels")

                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                self.original_img = img.copy()
                self.original_size = os.path.getsize(input_path) / 1024  # KB

            self.formats = ["PNG", "JPEG", "TIFF", "WebP"]
            self.quality_levels = [10, 30, 50, 70, 90]
            self.results = []
            self.converted_images = {}

            print(f"Image loaded successfully. Size: {width}x{height} pixels")

        def _convert_single_image(self, fmt, quality):
            """Convert a single image with given format and quality."""
            try:
                img_copy = self.original_img.copy()
                buffer = BytesIO()

                # Conversion timing
                start_time = time.time()
                if fmt == "TIFF":
                    img_copy.save(buffer, format=fmt, compression="tiff_lzw")
                else:
                    img_copy.save(buffer, format=fmt, quality=quality, optimize=True)
                conversion_time = time.time() - start_time

                # Get file size
                buffer.seek(0)
                file_size = len(buffer.getvalue()) / 1024  # Size in KB

                # Loading speed test
                load_buffer = BytesIO(buffer.getvalue())
                load_start = time.time()
                converted_img = Image.open(load_buffer)
                converted_img.load()
                loading_time = (time.time() - load_start) * 1000  # ms

                # Convert to numpy arrays for analysis
                original_array = np.array(self.original_img)
                converted_array = np.array(converted_img)

                # Calculate metrics
                mse = np.mean((original_array - converted_array) ** 2)
                psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")

                # Calculate SSIM with appropriate window size
                try:
                    min_dim = min(original_array.shape[0], original_array.shape[1])
                    win_size = min(
                        7, min_dim - (min_dim % 2) + 1
                    )  # Ensure odd window size
                    ssim_value = ssim(
                        original_array,
                        converted_array,
                        win_size=win_size,
                        channel_axis=-1,
                    )
                except Exception as e:
                    print(f"SSIM calculation failed: {str(e)}")
                    ssim_value = 0.0

                # Color analysis
                unique_colors = len(
                    np.unique(
                        converted_array.reshape(-1, converted_array.shape[-1]), axis=0
                    )
                )
                compression_ratio = self.original_size / file_size

                # Store converted image
                self.converted_images[(fmt, quality)] = converted_img

                return {
                    "format": fmt,
                    "quality_setting": quality,
                    "file_size": file_size,
                    "compression_ratio": compression_ratio,
                    "conversion_time": conversion_time * 1000,
                    "loading_time": loading_time,
                    "psnr": psnr,
                    "ssim": ssim_value,
                    "unique_colors": unique_colors,
                }

            except Exception as e:
                print(f"Error converting to {fmt} with quality {quality}: {str(e)}")
                return None

        def convert_images(self):
            """Convert images with optional parallel processing."""
            print("\nConverting images to different formats...")
            conversion_pairs = [
                (fmt, quality)
                for fmt in self.formats
                for quality in self.quality_levels
            ]

            if self.use_parallel:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for fmt, quality in conversion_pairs:
                        futures.append(
                            executor.submit(self._convert_single_image, fmt, quality)
                        )

                    for future in tqdm(
                        futures, total=len(conversion_pairs), desc="Processing formats"
                    ):
                        result = future.result()
                        if result is not None:
                            self.results.append(result)
            else:
                for fmt, quality in tqdm(conversion_pairs, desc="Processing formats"):
                    result = self._convert_single_image(fmt, quality)
                    if result is not None:
                        self.results.append(result)

        def _get_format_evaluation(self, format_data):
            """Generate dynamic format evaluation based on metrics."""
            avg_psnr = format_data["psnr"].mean()
            avg_ssim = format_data["ssim"].mean()
            avg_compression = format_data["compression_ratio"].mean()
            avg_loading = format_data["loading_time"].mean()

            evaluations = []

            # Quality evaluation
            if avg_psnr > 40 and avg_ssim > 0.95:
                quality_emoji = "✅"
                evaluations.append("Excellent quality retention")
            elif avg_psnr > 30 and avg_ssim > 0.90:
                quality_emoji = "✓"
                evaluations.append("Good quality retention")
            else:
                quality_emoji = "⚠️"
                evaluations.append("Noticeable quality loss")

            # Compression evaluation
            if avg_compression > 2:
                compression_emoji = "🏆"
                evaluations.append("Superior compression")
            elif avg_compression > 1:
                compression_emoji = "✓"
                evaluations.append("Decent compression")
            else:
                compression_emoji = "ℹ️"
                evaluations.append("Limited compression")

            # Speed evaluation
            if avg_loading < 50:
                speed_emoji = "⚡"
                evaluations.append("Fast loading")
            elif avg_loading < 100:
                speed_emoji = "✓"
                evaluations.append("Average loading speed")
            else:
                speed_emoji = "⏳"
                evaluations.append("Slower loading")

            emojis = f"{quality_emoji}{compression_emoji}{speed_emoji}"
            return emojis, " | ".join(evaluations)

        def get_detailed_analysis(self):
            """Generate a detailed analysis report with dynamic format evaluation."""
            df = pl.DataFrame(self.results)

            analysis = "\nDetailed Analysis Report:\n" + "=" * 50 + "\n"
            analysis += f"\nInput Image: {os.path.basename(self.input_path)}\n"
            analysis += (
                f"Original Format: {os.path.splitext(self.input_path)[1].upper()}\n"
            )
            analysis += f"Original Size: {self.original_size:.2f} KB\n\n"

            # Analyze each format
            for fmt in self.formats:
                format_data = df.filter(pl.col("format") == fmt)
                if not format_data.is_empty():
                    emojis, evaluation = self._get_format_evaluation(format_data)

                    analysis += f"\n{fmt} Format Analysis {emojis}\n{'-'*20}\n"
                    analysis += f"Performance: {evaluation}\n"
                    analysis += (
                        f"Average file size: {format_data['file_size'].mean():.2f} KB\n"
                    )
                    analysis += f"Compression ratio: {format_data['compression_ratio'].mean():.2f}x\n"
                    analysis += (
                        f"Best quality PSNR: {format_data['psnr'].max():.2f} dB\n"
                    )
                    analysis += f"Average SSIM: {format_data['ssim'].mean():.4f}\n"
                    analysis += (
                        f"Loading time: {format_data['loading_time'].mean():.2f} ms\n"
                    )

                    # Quality recommendations
                    best_quality = format_data.filter(
                        pl.col("ssim") == pl.col("ssim").max()
                    )
                    analysis += f"Recommended quality setting: {best_quality['quality_setting'][0]}\n"

            return analysis

        def analyze_bit_planes(self, image_array):
            """Analyze bit planes for steganography potential."""
            bit_planes = []
            scores = []

            # Extract and analyze each bit plane
            for bit in range(8):
                plane = (image_array & (1 << bit)) >> bit

                # Calculate randomness score (higher is better for hiding data)
                transitions = np.sum(np.abs(np.diff(plane))) / plane.size
                entropy = -np.sum(
                    (np.bincount(plane.flatten()) / plane.size)
                    * np.log2(np.bincount(plane.flatten()) / plane.size + 1e-10)
                )

                score = (transitions + entropy) / 2
                scores.append(score)
                bit_planes.append(plane)

            return scores, bit_planes

        def analyze_noise_pattern(self, image_array):
            """Analyze noise patterns that could hide data."""
            # Extract high-frequency components
            noise = image_array - gaussian_filter(image_array, sigma=2)

            # Calculate noise statistics
            noise_mean = np.mean(np.abs(noise))
            noise_std = np.std(noise)
            noise_entropy = -np.sum(
                (np.bincount(noise.flatten()) / noise.size)
                * np.log2(np.bincount(noise.flatten()) / noise.size + 1e-10)
            )

            return {
                "noise_level": noise_mean,
                "noise_variation": noise_std,
                "noise_entropy": noise_entropy,
            }

        def _get_stego_metrics(self, format_name):
            """Evaluate steganographic preservation metrics."""
            bit_plane_scores = []
            noise_metrics = []
            capacity_estimates = []

            for fmt, quality in self.converted_images.keys():
                if fmt == format_name:
                    img_array = np.array(self.converted_images[(fmt, quality)])

                    # Analyze bit planes
                    scores, _ = self.analyze_bit_planes(img_array)
                    bit_plane_scores.append(np.mean(scores))

                    # Analyze noise patterns
                    noise_data = self.analyze_noise_pattern(img_array)
                    noise_metrics.append(noise_data["noise_entropy"])

                    # Estimate potential steganographic capacity
                    capacity = (img_array.size * (quality / 100)) / 8  # bytes
                    capacity_estimates.append(capacity)

            return {
                "avg_bit_plane_score": np.mean(bit_plane_scores),
                "avg_noise_entropy": np.mean(noise_metrics),
                "avg_capacity": np.mean(capacity_estimates),
            }

        def get_stego_analysis(self):
            """Generate steganography-focused analysis report."""
            df = pl.DataFrame(self.results)
            unique_formats = df.select("format").unique().to_series().to_list()

            analysis = "\nSteganographic Analysis Report:\n" + "=" * 50 + "\n"
            analysis += "Evaluating formats for steganographic properties:\n\n"

            for fmt in unique_formats:
                stego_metrics = self._get_stego_metrics(fmt)

                # Evaluate steganographic potential
                if (
                    stego_metrics["avg_bit_plane_score"] > 0.7
                    and stego_metrics["avg_noise_entropy"] > 4
                ):
                    rating = "🔒 Excellent"
                elif (
                    stego_metrics["avg_bit_plane_score"] > 0.5
                    and stego_metrics["avg_noise_entropy"] > 3
                ):
                    rating = "🔐 Good"
                else:
                    rating = "⚠️ Limited"

                analysis += (
                    f"\n{fmt} Format Steganographic Properties {rating}\n{'-'*20}\n"
                )
                analysis += f"Bit-plane preservation: {stego_metrics['avg_bit_plane_score']:.3f}/1.0\n"
                analysis += (
                    f"Noise entropy: {stego_metrics['avg_noise_entropy']:.2f} bits\n"
                )
                analysis += f"Estimated max capacity: {stego_metrics['avg_capacity']/1024:.1f} KB\n"

                if fmt == "PNG":
                    analysis += "Like a perfect safe, preserves every detail for hiding data 🔒\n"
                elif fmt == "JPEG":
                    analysis += "Like a mosaic, some subtle patterns may be lost in compression 🎨\n"
                elif fmt == "WebP":
                    analysis += (
                        "Modern vault - good balance of security and efficiency 🏛️\n"
                    )
                elif fmt == "TIFF":
                    analysis += "Premium storage, keeps your secrets intact but takes more space 📦\n"

            return analysis

        def _get_timestamp_filename(self, base_name):
            """Generate filename with timestamp."""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{timestamp}_{base_name}"

        def save_figure(self, figure, base_name):
            """Save figure with timestamp in output directory."""
            filename = self._get_timestamp_filename(base_name)
            filepath = os.path.join(self.output_dir, filename)
            figure.savefig(filepath, bbox_inches="tight", dpi=300)
            print(f"Saved: {filepath}")
            return filepath

        def visualize_stego_effects(self):
            """Visualize how different formats affect steganographic content."""
            print("\nVisualizing steganography effects across formats...")

            # Create a figure with multiple rows for different analysis
            n_formats = len(self.formats)
            fig, axes = plt.subplots(4, n_formats, figsize=(5 * n_formats, 20))

            # Reference image (original)
            original_array = np.array(self.original_img)
            quality = 90  # Use high quality for comparison

            for idx, fmt in enumerate(self.formats):
                if (fmt, quality) in self.converted_images:
                    converted_array = np.array(self.converted_images[(fmt, quality)])

                    # Row 1: Show LSB plane (where steganographic data is often hidden)
                    lsb_original = original_array[:, :, 0] & 1
                    # lsb_converted = converted_array[:, :, 0] & 1

                    axes[0, idx].imshow(lsb_original, cmap="gray")
                    axes[0, idx].set_title(f"{fmt}\nLSB Layer")
                    axes[0, idx].axis("off")

                    # Row 2: Show high frequency noise pattern
                    noise_orig = original_array - gaussian_filter(
                        original_array, sigma=2
                    )
                    noise_conv = converted_array - gaussian_filter(
                        converted_array, sigma=2
                    )

                    # Normalize noise for visualization
                    noise_orig = np.clip(
                        (noise_orig - noise_orig.min())
                        / (noise_orig.max() - noise_orig.min()),
                        0,
                        1,
                    )
                    noise_conv = np.clip(
                        (noise_conv - noise_conv.min())
                        / (noise_conv.max() - noise_conv.min()),
                        0,
                        1,
                    )

                    # Show difference in noise patterns
                    noise_diff = np.abs(noise_orig - noise_conv)
                    axes[1, idx].imshow(np.mean(noise_diff, axis=2), cmap="hot")
                    axes[1, idx].set_title("Noise Pattern\nPreservation")
                    axes[1, idx].axis("off")

                    # Row 3: Bit plane patterns
                    bit_patterns = np.zeros_like(original_array[:, :, 0])
                    for bit in range(8):
                        bit_patterns += (
                            (converted_array[:, :, 0] & (1 << bit)) >> bit
                        ) * (2**bit)

                    axes[2, idx].imshow(bit_patterns, cmap="nipy_spectral")
                    axes[2, idx].set_title("Bit Plane\nDistribution")
                    axes[2, idx].axis("off")

                    # Row 4: Simulate hidden data survival
                    # Create a synthetic hidden pattern
                    h, w = original_array.shape[:2]
                    hidden_pattern = np.zeros((h, w), dtype=np.uint8)
                    # Add some text/pattern
                    text_pattern = "SECRET"
                    for i, char in enumerate(text_pattern):
                        start_x = w // 4 + i * 30
                        if start_x + 20 < w:
                            hidden_pattern[
                                h // 2 - 10 : h // 2 + 10, start_x : start_x + 20
                            ] = ord(char) % 2

                    # Embed and extract pattern
                    stego_img = original_array.copy()
                    stego_img[:, :, 0] = (stego_img[:, :, 0] & 254) | hidden_pattern

                    # Convert with the format
                    buffer = BytesIO()
                    Image.fromarray(stego_img).save(buffer, format=fmt, quality=quality)
                    buffer.seek(0)
                    converted_stego = np.array(Image.open(buffer))

                    # Extract and show the pattern
                    extracted_pattern = converted_stego[:, :, 0] & 1
                    pattern_diff = np.abs(extracted_pattern - hidden_pattern)

                    axes[3, idx].imshow(pattern_diff, cmap="RdYlGn_r")
                    axes[3, idx].set_title("Hidden Pattern\nSurvival")
                    axes[3, idx].axis("off")

                    # Add a text annotation showing preservation percentage
                    preservation = 100 * (1 - np.mean(pattern_diff))
                    axes[3, idx].text(
                        0.5,
                        -0.1,
                        f"{preservation:.1f}% preserved",
                        transform=axes[3, idx].transAxes,
                        ha="center",
                        va="center",
                    )

            plt.tight_layout()
            return fig

        def show_side_by_side_comparison(self, formats_to_compare=None, quality=70):
            """Show side-by-side image comparisons with zoomed regions and difference maps."""
            if formats_to_compare is None:
                formats_to_compare = self.formats

            n_formats = len(formats_to_compare)
            fig, axes = plt.subplots(3, n_formats, figsize=(5 * n_formats, 15))

            # Get interesting region to zoom (middle of the image)
            img_array = np.array(self.original_img)
            h, w = img_array.shape[:2]
            zoom_region = (slice(h // 3, 2 * h // 3), slice(w // 3, 2 * w // 3))

            # Use original image as reference
            reference_img = np.array(self.original_img)

            for idx, fmt in enumerate(formats_to_compare):
                if (fmt, quality) in self.converted_images:
                    current_img = np.array(self.converted_images[(fmt, quality)])

                    # Full image
                    if idx == 0:
                        axes[0, idx].imshow(reference_img)
                        axes[0, idx].set_title("Original Image (Reference)")
                    else:
                        axes[0, idx].imshow(current_img)
                        axes[0, idx].set_title(f"{fmt} Q{quality}")
                    axes[0, idx].axis("off")

                    # Zoomed region
                    if idx == 0:
                        axes[1, idx].imshow(reference_img[zoom_region])
                        axes[1, idx].set_title("Original (Zoomed)")
                    else:
                        axes[1, idx].imshow(current_img[zoom_region])
                        axes[1, idx].set_title(f"{fmt} Q{quality} (Zoomed)")
                    axes[1, idx].axis("off")

                    # Difference map
                    if idx > 0:
                        # Check if images are identical
                        is_identical = np.array_equal(current_img, reference_img)

                        if is_identical:
                            axes[2, idx].text(
                                0.5,
                                0.5,
                                "No Difference\n(Identical to Original)",
                                ha="center",
                                va="center",
                                fontsize=10,
                            )
                            axes[2, idx].set_title("Difference vs Original")
                        else:
                            diff = np.abs(
                                current_img.astype(float) - reference_img.astype(float)
                            )
                            diff = np.sum(diff, axis=2) / (3 * 255)
                            diff = np.clip(diff * 5, 0, 1)
                            im = axes[2, idx].imshow(diff, cmap="hot", vmin=0, vmax=1)
                            axes[2, idx].set_title("Difference vs Original")
                            plt.colorbar(im, ax=axes[2, idx])
                    else:
                        axes[2, idx].text(
                            0.5, 0.5, "Reference Image", ha="center", va="center"
                        )
                    axes[2, idx].axis("off")

            plt.tight_layout()
            return fig

    def main():
        input_image_path = "input_imgs/chng_mnd.jpg"
        output_dir = "output_imgs"

        print("\nInitializing Image Format Analyzer...")
        analyzer = ImageFormatAnalyzer(
            input_image_path, output_dir=output_dir, use_parallel=True, max_workers=8
        )

        # Convert images
        analyzer.convert_images()

        print("\nGenerating visualizations...")

        # Create and save side-by-side comparison
        comparison_fig = analyzer.show_side_by_side_comparison()
        analyzer.save_figure(comparison_fig, "side_by_side_comparison.png")
        plt.close(comparison_fig)

        # Print detailed analysis
        analysis = analyzer.get_detailed_analysis()
        print(analysis)

        # Print steganographic analysis
        stego_analysis = analyzer.get_stego_analysis()
        print(stego_analysis)

        # Generate and save steganography visualization
        stego_fig = analyzer.visualize_stego_effects()
        analyzer.save_figure(stego_fig, "stego_comparison.png")
        plt.close(stego_fig)

        # Save analysis to text file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_filename = f"{timestamp}_analysis_report.txt"
        analysis_filepath = os.path.join(output_dir, analysis_filename)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(analysis_filepath, "w", encoding="utf-8") as f:
            f.write(f"Analysis Report Generated: {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            f.write(analysis + "\n\n")
            f.write("-" * 50 + "\n\n")
            f.write(stego_analysis)
        print(f"Saved analysis report: {analysis_filepath}")

    if __name__ == "__main__":
        main()
    return (
        BytesIO,
        Image,
        ImageFormatAnalyzer,
        ThreadPoolExecutor,
        colorsys,
        compare_images,
        convolve,
        datetime,
        exposure,
        gaussian_filter,
        hsv_to_rgb,
        main,
        np,
        os,
        pl,
        plt,
        sns,
        ssim,
        time,
        tqdm,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
