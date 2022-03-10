import React from 'react';
import { render, screen } from '@testing-library/react';
import App from '../App';

test('renders application', () => {
  const { container } = render(<App />);
  const appName = container.firstChild;
  expect(appName?.textContent).toContain("React App");
});
