import React from "react";
import styles from './main-page.module.scss';

import Section
 from "../section/section";
function MainPage () {
    return (
        <div className={styles.container}>
            Hello!
            <Section 
                header="this is one section"
            />
        </div>
    )
}

export default MainPage;
