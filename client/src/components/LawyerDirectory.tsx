import { useState, useEffect } from "react";
import { Search, MapPin, Star, Briefcase, Loader2 } from "lucide-react";
import { Lawyer } from "../App";
import { fetchLawyers } from "../api";

interface Props {
  onSelectLawyer: (lawyer: Lawyer) => void;
  onGoToRecommendations: () => void;
}

export function LawyerDirectory({ onSelectLawyer, onGoToRecommendations }: Props) {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedSpecialty, setSelectedSpecialty] = useState("");
  const [sortBy, setSortBy] = useState<"rating" | "experience" | "price">("rating");
  const [lawyers, setLawyers] = useState<Lawyer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchLawyers()
      .then((data) => {
        setLawyers(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setError("Failed to load lawyers");
        setLoading(false);
      });
  }, []);

  const specialties = [
    "All Specialties",
    "Criminal Defense",
    "Family Law",
    "Business & Corporate Law",
    "Personal Injury",
    "Real Estate Law",
    "Immigration Law",
    "Intellectual Property",
    "Employment Law",
    "Tax Law",
  ];

  const filteredLawyers = lawyers
    .filter((lawyer) => {
      const matchesSearch =
        lawyer.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        lawyer.specialty.toLowerCase().includes(searchTerm.toLowerCase()) ||
        lawyer.location.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesSpecialty =
        !selectedSpecialty ||
        selectedSpecialty === "All Specialties" ||
        lawyer.specialty === selectedSpecialty;

      return matchesSearch && matchesSpecialty;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case "rating":
          return b.rating - a.rating;
        case "experience":
          return b.experience - a.experience;
        case "price":
          return a.hourlyRate - b.hourlyRate;
        default:
          return 0;
      }
    });

  if (loading) {
    return (
      <div className="flex justify-center items-center py-20">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-20 text-red-600">
        <p>{error}</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
        <div className="flex justify-between items-center mb-4">
          <h2 className="mb-4">Find a Lawyer</h2>
          <button
            onClick={onGoToRecommendations}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Smart Lawyer Finder
          </button>
        </div>

        {/* Search and Filters */}
        <div className="space-y-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchTerm(e.target.value)}
              placeholder="Search by name, specialty, or location..."
              className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
            />
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block mb-2 text-gray-700">Specialty</label>
              <select
                value={selectedSpecialty}
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setSelectedSpecialty(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
              >
                {specialties.map((specialty) => (
                  <option key={specialty} value={specialty}>
                    {specialty}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block mb-2 text-gray-700">Sort By</label>
              <select
                value={sortBy}
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setSortBy(e.target.value as "rating" | "experience" | "price")}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
              >
                <option value="rating">Highest Rating</option>
                <option value="experience">Most Experience</option>
                <option value="price">Lowest Price</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <p className="text-gray-600">
          {filteredLawyers.length} lawyer
          {filteredLawyers.length !== 1 ? "s" : ""} found
        </p>
      </div>

      {/* Lawyer Cards */}
      <div className="grid md:grid-cols-2 gap-6">
        {filteredLawyers.map((lawyer) => (
          <div
            key={lawyer.id}
            className="bg-white rounded-xl shadow-lg p-6 border border-gray-200 hover:shadow-xl transition-shadow"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <h3 className="mb-1">{lawyer.name}</h3>
                <p className="text-gray-600 mb-2">{lawyer.specialty}</p>
                <div className="flex items-center gap-4 text-gray-600">
                  <div className="flex items-center gap-1">
                    <Star className="w-4 h-4 text-yellow-500 fill-yellow-500" />
                    <span>{lawyer.rating}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Briefcase className="w-4 h-4" />
                    <span>{lawyer.experience} yrs</span>
                  </div>
                </div>
              </div>
              <div className="text-right">
                <p className="text-blue-600">${lawyer.hourlyRate}/hr</p>
              </div>
            </div>

            <p className="text-gray-700 mb-4 line-clamp-2">{lawyer.bio}</p>

            <div className="flex items-center justify-between pt-4 border-t border-gray-200">
              <div className="flex items-center gap-2 text-gray-600">
                <MapPin className="w-4 h-4" />
                <span>{lawyer.location}</span>
              </div>
              <button
                onClick={() => onSelectLawyer(lawyer)}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                View Profile
              </button>
            </div>
          </div>
        ))}
      </div>

      {filteredLawyers.length === 0 && (
        <div className="bg-white rounded-xl shadow-lg p-12 border border-gray-200 text-center">
          <p className="text-gray-600">
            No lawyers found matching your criteria. Try adjusting your search
            or filters.
          </p>
        </div>
      )}
    </div>
  );
}