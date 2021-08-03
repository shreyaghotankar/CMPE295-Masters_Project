import React from "react";
import Amplify from "aws-amplify";
import {AmplifyAuthenticator, AmplifySignIn, AmplifySignUp, AmplifySignOut} from "@aws-amplify/ui-react";
import awsconfig from "./aws-exports";

import MainPage from './components/main-page/main-page'

Amplify.configure(awsconfig);

const App = () => (
  <AmplifyAuthenticator usernameAlias="email">
      <AmplifySignUp
        slot="sign-up"
        usernameAlias="email"
        formFields={[
          {
            type: "email",
            label: "Email",
            placeholder: "Email",
            inputProps: { required: true, autocomplete: "username" },
          },
          {
            type: "password",
            label: "Password",
            placeholder: "********",
            inputProps: { required: true, autocomplete: "new-password" },
          },
          {
            type: "given_name",
            label: "Given Name",
            placeholder: "First Name",
            inputProps: { required: true, autocomplete: "given_name" },
          },
          {
            type: "family_name",
            label: "Family Name",
            placeholder: "Last Name",
            inputProps: { required: true, autocomplete: "family_name" },
          },
        ]} 
      />
      <AmplifySignIn slot="sign-in" usernameAlias="email" />
      <MainPage />
    </AmplifyAuthenticator>
);

export default App;
