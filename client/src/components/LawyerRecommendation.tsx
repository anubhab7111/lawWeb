import { useState } from 'react';
import { CheckCircle2, ArrowRight } from 'lucide-react';
import { Lawyer } from '../App';
import { mockLawyers } from './mockData';

interface Props {
  onSelectLawyer: (lawyer: Lawyer) => void;
}

interface FormData {
  legalIssue: string;
  urgency: string;
  budget: string;
  location: string;
}

export function LawyerRecommendation({ onSelectLawyer }: Props) {
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState<FormData>({
    legalIssue: '',
    urgency: '',
    budget: '',
    location: '',
  });
  const [recommendations, setRecommendations] = useState<Lawyer[]>([]);

  const handleInputChange = (field: keyof FormData, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = () => {
    // Filter lawyers based on form data
    let filtered = [...mockLawyers];

    if (formData.legalIssue) {
      filtered = filtered.filter(
        (lawyer) =>
          lawyer.specialty.toLowerCase().includes(formData.legalIssue.toLowerCase()) ||
          formData.legalIssue.toLowerCase().includes(lawyer.specialty.toLowerCase())
      );
    }

    if (formData.budget) {
      const maxBudget = parseInt(formData.budget);
      filtered = filtered.filter((lawyer) => lawyer.hourlyRate <= maxBudget);
    }

    if (formData.location) {
      filtered = filtered.filter((lawyer) =>
        lawyer.location.toLowerCase().includes(formData.location.toLowerCase())
      );
    }

    // Sort by rating and experience
    filtered.sort((a, b) => {
      if (b.rating !== a.rating) return b.rating - a.rating;
      return b.experience - a.experience;
    });

    setRecommendations(filtered.slice(0, 3));
    setStep(3);
  };

  return (
    <div className="max-w-4xl mx-auto">
      {step < 3 ? (
        <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
          <div className="mb-8">
            <h2 className="mb-2">Find Your Ideal Lawyer</h2>
            <p className="text-gray-600">
              Answer a few questions to get personalized lawyer recommendations
            </p>
          </div>

          {/* Progress indicator */}
          <div className="flex items-center justify-between mb-8">
            {[1, 2].map((s) => (
              <div key={s} className="flex items-center flex-1">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center ${
                    step >= s ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-600'
                  }`}
                >
                  {step > s ? <CheckCircle2 className="w-6 h-6" /> : s}
                </div>
                {s < 2 && (
                  <div
                    className={`flex-1 h-1 mx-2 ${
                      step > s ? 'bg-blue-600' : 'bg-gray-200'
                    }`}
                  />
                )}
              </div>
            ))}
          </div>

          {step === 1 && (
            <div className="space-y-6">
              <div>
                <label className="block mb-2 text-gray-700">
                  What type of legal issue do you have?
                </label>
                <select
                  value={formData.legalIssue}
                  onChange={(e) => handleInputChange('legalIssue', e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                >
                  <option value="">Select an issue...</option>
                  <option value="criminal">Criminal Defense</option>
                  <option value="family">Family Law (Divorce, Custody)</option>
                  <option value="business">Business & Corporate</option>
                  <option value="real estate">Real Estate</option>
                  <option value="personal injury">Personal Injury</option>
                  <option value="immigration">Immigration</option>
                  <option value="intellectual property">Intellectual Property</option>
                </select>
              </div>

              <div>
                <label className="block mb-2 text-gray-700">How urgent is your case?</label>
                <div className="grid grid-cols-3 gap-3">
                  {['Not urgent', 'Somewhat urgent', 'Very urgent'].map((urgency) => (
                    <button
                      key={urgency}
                      onClick={() => handleInputChange('urgency', urgency)}
                      className={`px-4 py-3 border rounded-lg transition-colors ${
                        formData.urgency === urgency
                          ? 'bg-blue-600 text-white border-blue-600'
                          : 'bg-white text-gray-700 border-gray-300 hover:border-blue-600'
                      }`}
                    >
                      {urgency}
                    </button>
                  ))}
                </div>
              </div>

              <button
                onClick={() => setStep(2)}
                disabled={!formData.legalIssue || !formData.urgency}
                className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                Next <ArrowRight className="w-5 h-5" />
              </button>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-6">
              <div>
                <label className="block mb-2 text-gray-700">
                  What is your budget range per hour?
                </label>
                <select
                  value={formData.budget}
                  onChange={(e) => handleInputChange('budget', e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                >
                  <option value="">Select budget...</option>
                  <option value="150">Under $150/hr</option>
                  <option value="250">$150 - $250/hr</option>
                  <option value="400">$250 - $400/hr</option>
                  <option value="1000">$400+/hr</option>
                </select>
              </div>

              <div>
                <label className="block mb-2 text-gray-700">Preferred location</label>
                <input
                  type="text"
                  value={formData.location}
                  onChange={(e) => handleInputChange('location', e.target.value)}
                  placeholder="Enter city or state..."
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                />
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => setStep(1)}
                  className="flex-1 px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                >
                  Back
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={!formData.budget}
                  className="flex-1 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  Get Recommendations <ArrowRight className="w-5 h-5" />
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
            <h2 className="mb-2">Recommended Lawyers for You</h2>
            <p className="text-gray-600">
              Based on your requirements, here are the best matches
            </p>
          </div>

          {recommendations.length === 0 ? (
            <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200 text-center">
              <p className="text-gray-600 mb-4">
                No lawyers found matching your criteria. Try adjusting your filters.
              </p>
              <button
                onClick={() => {
                  setStep(1);
                  setFormData({ legalIssue: '', urgency: '', budget: '', location: '' });
                }}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Start Over
              </button>
            </div>
          ) : (
            <div className="grid gap-6">
              {recommendations.map((lawyer, index) => (
                <div
                  key={lawyer.id}
                  className="bg-white rounded-xl shadow-lg p-6 border border-gray-200 hover:shadow-xl transition-shadow"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <h3>{lawyer.name}</h3>
                        {index === 0 && (
                          <span className="px-2 py-1 bg-blue-600 text-white rounded text-xs">
                            Top Match
                          </span>
                        )}
                      </div>
                      <p className="text-gray-600">{lawyer.specialty}</p>
                    </div>
                    <div className="text-right">
                      <div className="flex items-center gap-1 text-yellow-500 mb-1">
                        <span>★</span>
                        <span>{lawyer.rating}</span>
                      </div>
                      <p className="text-gray-600">{lawyer.experience} years exp.</p>
                    </div>
                  </div>

                  <p className="text-gray-700 mb-4">{lawyer.bio}</p>

                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <p className="text-gray-600">
                        <span className="text-gray-900">${lawyer.hourlyRate}/hr</span> • {lawyer.location}
                      </p>
                      <p className="text-gray-600">
                        {lawyer.cases} cases • {lawyer.successRate}% success rate
                      </p>
                    </div>
                    <button
                      onClick={() => onSelectLawyer(lawyer)}
                      className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    >
                      View Profile
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          <button
            onClick={() => {
              setStep(1);
              setFormData({ legalIssue: '', urgency: '', budget: '', location: '' });
            }}
            className="w-full px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
          >
            Start New Search
          </button>
        </div>
      )}
    </div>
  );
}
