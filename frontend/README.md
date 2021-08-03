## About React configurations

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

### Configuration: 
- node >= 10.16
- npm >= 5.6s
- node-sass [to use scss instead of plain css]
- prop-types [to typecheck]
- bootstrap

## Testing

Jest library
`npm test`

## Branching


 > ! Before you start always pull latest changes \
 > `git pull`

For each feature/bug or issue resolvment create a new branch 

`git checkout -b [branch name]`

Follow branch naming convention:
[feature/bug/issue]/[name of the reature]-[author]
Ex: feature/readme-az


## Amplify Setup

```
npm install -g @aws-amplify/cli
amplify upgrade
```

AWS best practices suggest to create a new user for each project and grant granual permissions to it. 

After the user is create in AWS console and creadentials for CLI (access key Id and secret access key) are recieved, aws cli should be cofigured on a local machine. 

Run the command to create a new profile and you will be prompted to add credentials: 
```
aws configure --profile [new_profile_name]
```

Output: 
```
AWS Access Key ID [None]: SOMEKEYID
AWS Secret Access Key [None]: SoMeSeCrEtHeRe
Default region name [None]: us-west-2 // or any other region of your choice 
Default output format [None]: json // or any other format
```

To avoid specifying the profile in every command set the AWS_PROFILE environment variable: 
```
export AWS_PROFILE=master_project
```
##  Project Folder Structure
-> components (holds all components by separate folders) \
---> component-name1 \
---> component-name2 \
... \
-> shared \
---> icons (holds all icons for the project) \
---> styles (holds all scss files for global styles and variables) 

### Component Folder Structure

*Folder name*: component-name
*Files*:
- component-name.js
- component-name.test.js
- component-name.module.scss

**component-name.js**
```javascript
import React from "react";
import PropTypes from 'prop-types'; // for typechecking

import styles from './component-name.module.scss'; // for creating unique local class names

function ComponentName (props) {
    const {
        propOne,
        propTwo
    } = props;

    return (
        <div>
            {propOne}
        </div>
    )
}

ComponentName.propTypes = {
    propOne: PropTypes.string, //type of the property should be specified
    propTwo: PropTypes.bool
}

export default ComponentName;

```


## Styles

To destinguish local vs global styles we are using Css Modules that automatically create a unique classname in the format `[filename]\_[classname]\_\_[hash]`

## References:

[Configure AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) \
[CSS Modules](https://create-react-app.dev/docs/adding-a-css-modules-stylesheet)
