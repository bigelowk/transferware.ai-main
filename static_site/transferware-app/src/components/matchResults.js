import React from "react";
import { useData } from "../DataContext";

const MatchResults = () => {
  const { data } = useData();

  if (!data) return <p>No data available</p>;

  return (
    <div className="flex items-center lg:p-4">
      <div className="flex grid grid-cols-2 lg:grid-cols-3 gap-2 p-4 w-full">
        {data.map((item) => (
          <a
            key={item.id}
            className="flex flex-col justify-center p-3 hover:border hover:shadow-sm"
            href={item.tcc_url}
            target="_blank"
          >
            <div className="flex flex-row items-center lg:min-w-60 sm:p-6">
              <img
                src={item.imageUrl}
                alt="Pattern-img"
                className="w-full lg:w-full mb-2"
              />
            </div>
            <p className="mb-1 font-serif text-xl font-semibold">
              <span className="font-semibold "></span> {item.pattern_name}
            </p>
            <p className="mb-4 font-light text-gray-700">
              <span className="">Confidence:</span> {item.confidence.toFixed(3)}
            </p>
          </a>
        ))}
      </div>
    </div>
  );
};

export default MatchResults;

