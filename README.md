[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/) [![Windows](https://img.shields.io/badge/Platform-Windows-green.svg)]()
# Image Format Comparison Tool 🖼️

This is just a little tool for comparing different image formats. It analyzes JPEG, TIFF, WebP, and PNG formats, looking at things like how well they compress, how quickly they load, their quality, and their potential for hiding data (steganography).

## What's Coming Next? 🚀

- [ ] Working on turning this into an interactive website for easier use
- [ ] Planning to add exciting JPEG XL support for even more comparison options
- [x] Added the initial Python script
- [x] Added setup file for easy installation

## How It Works 📊

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#6366f1',
    'primaryTextColor': '#fff',
    'primaryBorderColor': '#4f46e5',
    'lineColor': '#6366f1',
    'secondaryColor': '#f0f9ff',
    'tertiaryColor': '#e0e7ff'
  }
}}%%
flowchart TD
    Input[/Input Image/]:::input --> Convert[Convert to Multiple Formats]
    Convert --> Analyze[Analyze Images]
    Analyze -->|Quality Metrics| Report[Generate Report]:::output
    Analyze -->|Visual Compare| Visualize[Create Visualizations]:::output

    subgraph Formats[Image Formats]
        direction LR
        PNG:::format
        JPEG:::format
        TIFF:::format
        WebP:::format
    end

    Convert --> Formats
    classDef default fill:#6366f1,stroke:#4f46e5,stroke-width:2px,color:#fff,rounded:true
    classDef input fill:#22c55e,stroke:#16a34a,stroke-width:2px,color:#fff,rounded:true
    classDef output fill:#ec4899,stroke:#db2777,stroke-width:2px,color:#fff,rounded:true
    classDef format fill:#f0f9ff,stroke:#6366f1,stroke-width:2px,color:#6366f1,rounded:true
```

## Getting Started 🚀

1. First, clone this repository to your computer
2. Run the `setup.bat` file (compatible with Windows 10 and Python 3.11)
3. Launch the tool by typing `marimo edit` in your terminal - it'll open right in your browser!
4. Run the `test_jxl.py` file to see how it works

## License & Usage 📝

This tool is for personal use - feel free to use it for your own projects and learning! Not intended for commercial applications.

## Want to Learn More? 📚

Check out [JPEG XL](https://jpegxl.info/index.html) to learn about this exciting new image format we'll be adding soon!
