import React, { useState, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import photoIcon from "../assets/images/photo-icon.png";
import { useData } from "../DataContext";
import heic2any from "heic2any";
import Footer from "../components/footer";

function UploadPage() {
  const location = useLocation();
  const [selectedFile, setSelectedFile] = useState(null);
  const [errorMessage, setErrorMessage] = useState(
    (location.state || { errorMessage: "" }).errorMessage
  );
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const [fileSize, setFileSize] = useState("");
  const fileInputRef = useRef(null);
  const { setData } = useData();
  const navigate = useNavigate();

  // Base url for the query API
  const base_url = process.env.REACT_APP_QUERY_BASE;

  const handleFileChange = async (file) => {
    if (
      file &&
      (file.type === "image/png" ||
        file.type === "image/jpeg" ||
        file.type === "image/heic")
    ) {
      let processedFile = file;

      // Convert HEIC to JPEG if needed
      if (file.type === "image/heic") {
        try {
          const convertedBlob = await heic2any({
            blob: file,
            toType: "image/jpeg",
          });
          processedFile = new File(
            [convertedBlob],
            file.name.replace(/\.heic$/i, ".jpg"),
            { type: "image/jpeg" }
          );
        } catch (error) {
          setErrorMessage("Failed to convert HEIC to JPEG.");
          return;
        }
      }

      setSelectedFile(processedFile);
      setErrorMessage("");
      setUploadedFileName(processedFile.name);
      setImagePreviewUrl(URL.createObjectURL(processedFile));

      // Calculate file size in kilobytes
      const fileSizeInKB = Math.round((processedFile.size / 1024) * 100) / 100;
      setFileSize(fileSizeInKB + " KB");
    } else {
      setSelectedFile(null);
      setUploadedFileName("");
      setFileSize("");
      setImagePreviewUrl(null);
      setErrorMessage("Please select a valid PNG, JPG, or HEIC file.");
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    handleFileChange(file);
  };

  const handleUrlChange = (e) => {
    setImageUrl(e.target.value);
  };

  const handleImportFromUrl = async () => {
    if (imageUrl.trim() !== "") {
      try {
        // Fetch the image from the URL as a Blob
        const response = await fetch(imageUrl);
        const imageBlob = await response.blob();

        // Create a filename from URL
        let filename = imageUrl.substring(imageUrl.lastIndexOf("/") + 1);
        const queryIndex = filename.indexOf("?");
        if (queryIndex !== -1) {
          filename = filename.substring(0, queryIndex);
        }

        // Create a File object from the Blob
        const imageFile = new File([imageBlob], filename, {
          type: imageBlob.type,
        });

        setSelectedFile(imageFile);
        setUploadedFileName(filename);
        setFileSize((imageBlob.size / 1024).toFixed(2) + " KB");
        setImagePreviewUrl(URL.createObjectURL(imageBlob));
        setErrorMessage("");
        setImageUrl("");
      } catch (error) {
        console.error("Error loading image:", error);
        setErrorMessage("Error: Unable to load image from URL.");
      }
    } else {
      setErrorMessage("Please enter a valid image URL.");
    }
  };

  const handleCancel = () => {
    setSelectedFile(null);
    setUploadedFileName("");
    setFileSize("");
    setImagePreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSubmit = async () => {
    if (selectedFile) {
      navigate("/loading");
      const formData = new FormData();
      formData.append("file", selectedFile);

      try {
        const response = await fetch(`${base_url}/query`, {
          method: "POST",
          body: formData,
        });

        const queryResults = await response.json();

        const patternFetchPromises = queryResults.map(async (result) => {
          const patternResponse = await fetch(
            `${base_url}/pattern/${result.id}`
          );
          const patternData = await patternResponse.json();
          return {
            id: result.id,
            confidence: result.confidence,
            pattern_name: patternData.pattern_name,
            tcc_url: patternData.tcc_url,
          };
        });

        const combinedResults = await Promise.all(patternFetchPromises);

        const patternImagePromises = combinedResults.map(async (pattern) => {
          try {
            const imageUrlResponse = await fetch(
              `${base_url}/pattern/image/${pattern.id}`
            );
            if (!imageUrlResponse.ok) throw new Error("Failed to load image");
            const imageUrl = await imageUrlResponse.url;
            return {
              ...pattern,
              imageUrl: imageUrl,
            };
          } catch {
            return {
              ...pattern,
              imageUrl:
                "https://cdn.vectorstock.com/i/500p/36/49/no-image-symbol-missing-available-icon-gallery-vector-43193649.jpg",
            };
          }
        });

        const finalResults = await Promise.all(patternImagePromises);
        setData(finalResults);

        navigate("/viewMatches", {
          replace: true,
          state: { imagePreviewUrl: imagePreviewUrl },
        });
      } catch (error) {
        navigate("/uploadPage", { state: { errorMessage: "Failed to submit the file." } });
        console.error("Error fetching data:", error);
      }
    }
  };

  return (
    <div>
      <div className="flex justify-center items-center w-full p-10 h-screen pt-16">
        <div className="flex bg-white flex-col items-center md:w-3/5 w-full px-8 sm:px-20 py-4 rounded-xl border-2 border-slate-400 shadow-[5px_5px_rgba(100,_116,_139,_0.4),_10px_10px_rgba(100,_116,_139,_0.3),_15px_15px_rgba(100,_116,_139,_0.2),_20px_20px_rgba(100,_116,_139,_0.1),_25px_25px_rgba(100,_116,_139,_0.05)]">
          <h1 className="w-full pb-5 text-center sm:text-start font-semibold text-xl">
            Upload a Photo of Your Sherd
          </h1>
          <div
            className="w-full h-1/3 flex flex-col items-center justify-center border-2 border-slate-300 border-dashed rounded-md py-8 lg:py-16 px-6"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <img src={photoIcon} className="max-h-28" alt="photo-icon" />
            <h2 className="text-center font-medium py-0">
              Drop your image here, or
              <label
                htmlFor="fileInput"
                className="cursor-pointer font-bold text-blue-600"
              >
                {" "}
                browse
              </label>
              <input
                ref={fileInputRef}
                id="fileInput"
                type="file"
                accept=".png,.jpg,.jpeg,.heic"
                onChange={(e) => handleFileChange(e.target.files[0])}
                className="hidden"
              />
            </h2>
            <p className="text-neutral-400 text-xs py-2 sm:py-0">
              Supports: PNG, JPG, and HEIC images
            </p>
            {errorMessage && <p className="text-red-500">{errorMessage}</p>}
          </div>

          {/* Uploaded file name & size display */}
          <div
            className={`w-full flex flex-row items-center border-2 rounded-lg px-4 sm:px-8 py-1 mt-6 ${
              !uploadedFileName && "hidden"
            }`}
          >
            <img
              src={imagePreviewUrl}
              alt="Preview picture"
              className="max-h-16 rounded-lg"
            />
            {uploadedFileName && (
              <p className="w-full px-2 sm:px-8 py-4 flex flex-col justify-between break-words break-all text-xs font-semibold text-blue-900">
                {uploadedFileName}
                <span className="pt-2 text-zinc-400 font-semibold">
                  ({fileSize})
                </span>
              </p>
            )}
          </div>

          <div className="relative flex py-5 w-full items-center">
            <div className="flex-grow border-t border-gray-400"></div>
            <span className="flex-shrink mx-4 text-gray-400">or</span>
            <div className="flex-grow border-t border-gray-400"></div>
          </div>
          <div className="w-full">
            <div className="p-4">Import from URL</div>
            <div className="flex justify-between bg-gray-100 rounded-md">
              <input
                type="text"
                placeholder="Enter image URL"
                value={imageUrl}
                onChange={handleUrlChange}
                className="flex-grow mr-2 bg-gray-100 px-6 py-4 w-full rounded-md"
              />
              <button
                onClick={handleImportFromUrl}
                className=" text-gray-500 px-4 py-2 font-semibold"
              >
                Upload
              </button>
            </div>

            <div className="flex justify-center mt-6 sm:justify-end space-x-4">
              <button
                className={`border-2 border-gray-400 text-black font-semibold px-4 py-2 rounded-md ${
                  selectedFile ? "" : "opacity-60 cursor-not-allowed"
                }`}
                disabled={!selectedFile}
                onClick={handleCancel}
              >
                Cancel
              </button>
              <button
                className={`bg-black text-white font-semibold px-4 py-2 rounded-md ${
                  selectedFile ? "" : "bg-opacity-70 cursor-not-allowed"
                }`}
                disabled={!selectedFile}
                onClick={handleSubmit}
              >
                Submit
              </button>
            </div>
          </div>
        </div>
      </div>
      <Footer/>
    </div>
  );
}

export default UploadPage;
