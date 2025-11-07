/**
 * Loading Spinner Component
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 15:32:27 UTC
 */

const LoadingSpinner = ({ size = 'medium', className = '' }) => {
  const sizes = {
    small: 'w-4 h-4 border-2',
    medium: 'w-8 h-8 border-3',
    large: 'w-12 h-12 border-4',
  };

  return (
    <div className={`inline-block ${className}`}>
      <div
        className={`${sizes[size]} border-gray-300 border-t-red-600 rounded-full animate-spin`}
      ></div>
    </div>
  );
};

export default LoadingSpinner;
