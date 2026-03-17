import { ArrowLeft, Star, MapPin, Briefcase, GraduationCap, Languages, Calendar, Award, TrendingUp } from 'lucide-react';
import { Lawyer } from '../App';

interface Props {
  lawyer: Lawyer;
  onBook: (lawyer: Lawyer) => void;
  onBack: () => void;
}

export function LawyerProfile({ lawyer, onBook, onBack }: Props) {
  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Back Button */}
      <button
        onClick={onBack}
        className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors"
      >
        <ArrowLeft className="w-5 h-5" />
        Back to directory
      </button>

      {/* Main Profile Card */}
      <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
        <div className="flex items-start justify-between mb-6">
          <div className="flex-1">
            <h1 className="mb-2">{lawyer.name}</h1>
            <p className="text-gray-600 mb-3">{lawyer.specialty}</p>
            <div className="flex items-center gap-6 text-gray-600">
              <div className="flex items-center gap-2">
                <Star className="w-5 h-5 text-yellow-500 fill-yellow-500" />
                <span>{lawyer.rating} Rating</span>
              </div>
              <div className="flex items-center gap-2">
                <Briefcase className="w-5 h-5" />
                <span>{lawyer.experience} Years Experience</span>
              </div>
              <div className="flex items-center gap-2">
                <MapPin className="w-5 h-5" />
                <span>{lawyer.location}</span>
              </div>
            </div>
          </div>
          <div className="text-right">
            <p className="text-gray-600 mb-1">Hourly Rate</p>
            <p className="text-blue-600">${lawyer.hourlyRate}/hour</p>
          </div>
        </div>

        <div className="flex items-center gap-4 p-4 bg-blue-50 rounded-lg mb-6">
          <Calendar className="w-6 h-6 text-blue-600" />
          <div>
            <p className="text-blue-900">Availability</p>
            <p className="text-blue-700">{lawyer.availability}</p>
          </div>
        </div>

        <button
          onClick={() => onBook(lawyer)}
          className="w-full px-6 py-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Book Consultation
        </button>
      </div>

      {/* About Section */}
      <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
        <h2 className="mb-4">About</h2>
        <p className="text-gray-700 leading-relaxed">{lawyer.bio}</p>
      </div>

      {/* Stats Grid */}
      <div className="grid md:grid-cols-3 gap-6">
        <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
              <Briefcase className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <p className="text-gray-600">Cases Handled</p>
              <p className="text-gray-900">{lawyer.cases}+</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <p className="text-gray-600">Success Rate</p>
              <p className="text-gray-900">{lawyer.successRate}%</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center">
              <Award className="w-6 h-6 text-yellow-600" />
            </div>
            <div>
              <p className="text-gray-600">Client Rating</p>
              <p className="text-gray-900">{lawyer.rating}/5.0</p>
            </div>
          </div>
        </div>
      </div>

      {/* Details Grid */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Education */}
        <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
          <div className="flex items-center gap-2 mb-4">
            <GraduationCap className="w-6 h-6 text-blue-600" />
            <h3>Education</h3>
          </div>
          <p className="text-gray-700">{lawyer.education}</p>
        </div>

        {/* Languages */}
        <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
          <div className="flex items-center gap-2 mb-4">
            <Languages className="w-6 h-6 text-blue-600" />
            <h3>Languages</h3>
          </div>
          <div className="flex flex-wrap gap-2">
            {lawyer.languages.map((language) => (
              <span
                key={language}
                className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full"
              >
                {language}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Practice Areas */}
      <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
        <h3 className="mb-4">Practice Areas</h3>
        <div className="flex flex-wrap gap-2">
          <span className="px-4 py-2 bg-blue-100 text-blue-700 rounded-lg">
            {lawyer.specialty}
          </span>
        </div>
      </div>

      {/* Bottom CTA */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-xl shadow-lg p-8">
        <div className="text-center">
          <h2 className="mb-2 text-white">Ready to get started?</h2>
          <p className="mb-6 text-blue-100">
            Book a consultation with {lawyer.name.split(' ')[0]} today
          </p>
          <button
            onClick={() => onBook(lawyer)}
            className="px-8 py-4 bg-white text-blue-600 rounded-lg hover:bg-gray-100 transition-colors"
          >
            Schedule Consultation
          </button>
        </div>
      </div>
    </div>
  );
}
